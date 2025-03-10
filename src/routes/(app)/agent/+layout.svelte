<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { WEBUI_NAME, showSidebar, functions } from '$lib/stores';
	import MenuLines from '$lib/components/icons/MenuLines.svelte';
	import { page } from '$app/stores';
	import Sidebar from '$lib/components/layout/Sidebar.svelte';
	import SettingsModal from '$lib/components/chat/SettingsModal.svelte';
	const i18n = getContext('i18n');

	onMount(async () => {});
</script>

<svelte:head>
	<title>
		{$i18n.t('Agent')} | {$WEBUI_NAME}
	</title>
</svelte:head>


<slot />
