<script lang="ts">
  import PaperExplorable from '$lib/templates/PaperExplorable.svelte';
  import type { PageData } from './$types';

  export let data: PageData;
</script>

{#if data.paper}
  <PaperExplorable paper={data.paper} />
{:else}
  <div class="error">
    <h1>Paper not found</h1>
    <p>The requested paper could not be loaded.</p>
    <a href="/">← Back to all papers</a>
  </div>
{/if}

<style>
  .error {
    max-width: 600px;
    margin: 4rem auto;
    text-align: center;
    font-family: 'Inter', sans-serif;
  }

  .error h1 {
    font-size: 2rem;
    color: #1f2937;
    margin-bottom: 1rem;
  }

  .error p {
    color: #6b7280;
    margin-bottom: 2rem;
  }

  .error a {
    color: #3b82f6;
    text-decoration: none;
  }

  .error a:hover {
    text-decoration: underline;
  }
</style>
