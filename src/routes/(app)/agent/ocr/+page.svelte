<script lang="ts">
    import { onMount, getContext } from 'svelte'; 
    import Navbar from '$lib/components/chat/Navbar.svelte'
    import Chat from '$lib/components/chat/Chat.svelte'
    import Chem from '$lib/components/agent/chem.svelte';
    import UserMenu from '$lib/components/layout/Sidebar/UserMenu.svelte';
    import { WEBUI_NAME, type Model, models, chatId, chatTitle, settings } from '$lib/stores';

    const i18n = getContext('i18n');
    let params = {};
    onMount(async () => {
    });
</script>

<svelte:head>
	<title>
		{$i18n.t('OCR')} | {$WEBUI_NAME}
	</title>
</svelte:head>

<Chat />