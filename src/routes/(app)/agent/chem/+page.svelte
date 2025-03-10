<script lang="ts">
    import { onMount, getContext } from 'svelte'; 
    import Navbar from '$lib/components/chat/Navbar.svelte'
    import Chat from '$lib/components/chat/Chat.svelte'
    import Chem from '$lib/components/agent/chem.svelte';
    import UserMenu from '$lib/components/layout/Sidebar/UserMenu.svelte';
    import { WEBUI_NAME, type Model, models, chatId, chatTitle, settings } from '$lib/stores';
    
    const i18n = getContext('i18n');
    let title = "化工大模型"
    let navbarElement;
    let selectedModels = ['chem:latest'];
	let atSelectedModel: Model | undefined;
	let selectedModelIds = [];
	$: selectedModelIds = atSelectedModel !== undefined ? [atSelectedModel.id] : selectedModels;
    let params = {};
    onMount(async () => {
    });
</script>
<svelte:head>
	<title>
		{$i18n.t('Chem')} | {$WEBUI_NAME}
	</title>
</svelte:head>

<Chat chatTitle="化工大模型" />

<!-- <Chem /> -->
