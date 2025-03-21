import os
import logging
import base64
import json
import time
import asyncio
import uuid
import numpy as np
import cv2
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from open_webui.utils.auth import get_verified_user
from open_webui.config import CACHE_DIR
from open_webui.env import SRC_LOG_LEVELS

# 设置日志
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MAIN", logging.INFO))

# 创建OCR上传目录
OCR_UPLOAD_DIR = CACHE_DIR / "ocr_uploads"
OCR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 定义数据模型
class OCRResult(BaseModel):
    text: str
    confidence: float
    box: Optional[List[List[int]]] = None

class OCRResponse(BaseModel):
    results: List[OCRResult]
    processing_time: float

# 创建路由
router = APIRouter()

class CustomOCRModel:
    """自定义OCR模型封装类"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.model_loaded = False
        self.is_loading = False
        self.lock = asyncio.Lock()

    async def initialize_model(self):
        """初始化模型（懒加载）"""
        if self.model_loaded or self.is_loading:
            return
        
        async with self.lock:
            if self.model_loaded or self.is_loading:
                return
            
            self.is_loading = True
            try:
                log.info("正在加载OCR模型...")
                
                # 在后台线程中加载模型，避免阻塞
                await asyncio.to_thread(self._load_model)
                
                self.model_loaded = True
                log.info("OCR模型加载完成")
            except Exception as e:
                log.error(f"模型加载错误: {str(e)}")
            finally:
                self.is_loading = False
    
    def _load_model(self):
        """实际加载模型的方法（在单独线程中执行）"""
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            
            # 检查CUDA可用性
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 加载模型和处理器
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "allenai/olmOCR-7B-0225-preview", 
                torch_dtype=torch.bfloat16
            ).eval().to(self.device)
            
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            
        except Exception as e:
            log.error(f"模型加载错误: {str(e)}")
            raise e
    
    async def process_image(self, image_data, region=None):
        """处理图像并返回OCR结果"""
        start_time = time.time()
        
        try:
            # 转换图像为OpenCV格式
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_data
            
            # 如果指定了区域，裁剪图像
            if region:
                x, y, w, h = region
                image = image[y:y+h, x:x+w]
            
            # 处理图像
            if self.model_loaded:
                results = await asyncio.to_thread(self._process_with_olm_ocr, image)
            else:
                # 如果模型未加载，返回占位结果
                results = self._get_dummy_results(image)
            
            # 如果使用了区域，调整坐标
            if region and results:
                x_offset, y_offset = region[0], region[1]
                for result in results:
                    if result.box:
                        result.box = [
                            [point[0] + x_offset, point[1] + y_offset] 
                            for point in result.box
                        ]
            
            processing_time = time.time() - start_time
            return OCRResponse(results=results, processing_time=processing_time)
            
        except Exception as e:
            log.error(f"OCR处理错误: {str(e)}")
            processing_time = time.time() - start_time
            return OCRResponse(results=[], processing_time=processing_time)
    
    def _process_with_olm_ocr(self, image):
        """使用OlmOCR处理图像"""
        try:
            import torch
            import base64
            from PIL import Image
            from io import BytesIO
            from olmocr.prompts import build_finetuning_prompt
            
            # 转换OpenCV图像为PIL格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 准备输入
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 构建提示
            prompt = build_finetuning_prompt("")
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 处理输入
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for (key, value) in inputs.items()}
            
            # 生成输出
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    temperature=0.8,
                    max_new_tokens=1024,  # 允许更长的输出
                    num_return_sequences=1,
                    do_sample=True,
                )
            
            # 解码输出
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text_output = self.processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0]
            
            log.info(f"OlmOCR输出: {text_output}")
            
            # 解析JSON响应
            try:
                result_json = json.loads(text_output)
                recognized_text = result_json.get("natural_text", "")
                
                # 创建结果
                ocr_result = OCRResult(
                    text=recognized_text,
                    confidence=0.9,  # OlmOCR没有置信度，给一个默认值
                    box=None  # OlmOCR通常不返回边界框
                )
                
                return [ocr_result]
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接使用文本
                return [OCRResult(text=text_output, confidence=0.8, box=None)]
                
        except Exception as e:
            log.error(f"OlmOCR处理错误: {str(e)}")
            return self._get_dummy_results(image)
    
    def _get_dummy_results(self, image):
        """返回占位OCR结果（当模型未加载或处理失败时）"""
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        return [
            OCRResult(
                text="OCR模型未能加载或处理失败",
                confidence=0.1,
                box=[[10, 10], [width-10, 10], [width-10, height-10], [10, height-10]]
            )
        ]

    async def process_pdf(self, pdf_data):
        """处理PDF文件并返回OCR结果"""
        try:
            # 创建临时文件保存PDF
            pdf_path = OCR_UPLOAD_DIR / f"{uuid.uuid4()}.pdf"
            
            with open(pdf_path, "wb") as f:
                f.write(pdf_data)
            
            # 使用PyMuPDF渲染PDF页面
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            all_results = []
            total_time = 0  # 记录总处理时间
            
            for page_num in range(min(doc.page_count, 5)):  # 最多处理前5页
                page = doc.load_page(page_num)
                
                # 渲染页面为图像
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x缩放以提高清晰度
                img_data = pix.tobytes("png")
                
                # 处理图像
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 进行OCR
                page_start_time = time.time()
                page_results = await self.process_image(image)
                page_time = time.time() - page_start_time
                total_time += page_time
                
                # 添加页码信息
                for result in page_results.results:
                    result_dict = result.dict()
                    result_dict["page"] = page_num + 1
                    all_results.append(OCRResult(**result_dict))
            
            # 清理临时文件
            os.unlink(pdf_path)
            
            return OCRResponse(
                results=all_results, 
                processing_time=total_time  # 使用累计的总时间
            )
            
        except Exception as e:
            log.error(f"PDF OCR处理错误: {str(e)} - {locals().get('page_results', {})}")
            return OCRResponse(results=[], processing_time=0)

# 实例化模型
ocr_model = CustomOCRModel()

@router.post("/api/v1/ocr", response_model=OCRResponse)
async def process_ocr(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    # user = Depends(get_verified_user)  # 注释掉这行，移除身份认证要求
):
    """上传图像或PDF进行OCR识别"""
    try:
        # 读取文件内容
        content = await file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # 后台启动模型加载（如果尚未加载）
        background_tasks.add_task(ocr_model.initialize_model)
        
        # 根据文件类型处理
        if file_extension in ['.pdf']:
            return await ocr_model.process_pdf(content)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return await ocr_model.process_image(content)
        else:
            raise HTTPException(status_code=400, detail="不支持的文件类型")
    
    except Exception as e:
        log.error(f"OCR处理请求错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR处理错误: {str(e)}")

@router.post("/api/v1/ocr/region", response_model=OCRResponse)
async def process_ocr_region(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    background_tasks: BackgroundTasks = None,
    # user = Depends(get_verified_user)  # 注释掉这行，移除身份认证要求
):
    """对图像的指定区域进行OCR识别"""
    try:
        # 读取文件内容
        content = await file.read()
        
        # 后台启动模型加载（如果尚未加载）
        background_tasks.add_task(ocr_model.initialize_model)
        
        # 检查坐标有效性
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="无效的区域坐标")
        
        # 处理指定区域
        region = (x, y, width, height)
        return await ocr_model.process_image(content, region)
    
    except Exception as e:
        log.error(f"区域OCR处理请求错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR处理错误: {str(e)}") 