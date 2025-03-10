<script lang="ts">
	import { marked } from 'marked';

	import { toast } from 'svelte-sonner';
	import Sortable from 'sortablejs';

	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;

	import { onMount, getContext, tick } from 'svelte';
	import { goto } from '$app/navigation';
	const i18n = getContext('i18n');

	import { WEBUI_NAME, config, mobile, models as _models, settings, user } from '$lib/stores';
    import Spinner from '../common/Spinner.svelte';
    let loaded = false;

    onMount(async () => {
        loaded = true;
    })
</script>

<svelte:head>
	<title>
		{$i18n.t('Chem')} | {$WEBUI_NAME}
	</title>
</svelte:head>

{#if loaded}
    <div class="w-full h-full flex justify-center items-center">
        化工大模型对话界面
    </div>
{:else}
    <div class="w-full h-full flex justify-center items-center">
        <Spinner />
    </div>
{/if}
