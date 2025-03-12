# 本地开发设置

## 前端

```bash
cd open-webui
npm install
npm run dev
```

## 后端

```bash
cd backend
conda create --name open-webui python=3.11
conda activate open-webui
pip install -r requirements.txt -U
sh dev.sh
```

## 配置

### 跨域问题处理
.env文件9行，IP修改为实际运行位置IP

### VS Code
.svelte文件关联语言Svelte，减少报错，安装JS、TS和ESLint相关插件

# 前端开发

- src
    + routes 路由
        - (app) 默认路由"/"
            + agent 新添加路由
                - chem 可跳转/agent/chem
                - ocr 可跳转/agent/ocr
                - +layout.svelte 设置/agent页面布局
- lib
    + camponents 组件
        - agent 新添加页面
            + chem.svelte 化工大模型页面
            + ocr.svelte OCR页面

# 后端开发
待补充