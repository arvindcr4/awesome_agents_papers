<script lang="ts">
  /**
   * Distill-style collapsible section for progressive disclosure.
   * Usage: <Collapsible title="Show more details">Advanced content</Collapsible>
   */
  export let title: string = 'More details';
  export let open: boolean = false;
</script>

<details class="collapsible" bind:open>
  <summary>{title}</summary>
  <div class="content">
    <slot />
  </div>
</details>

<style>
  .collapsible {
    margin: 1.5rem 0;
    padding: 1rem;
    background: #f9fafb;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
  }

  summary {
    cursor: pointer;
    font-weight: 500;
    color: #374151;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  summary::-webkit-details-marker {
    display: none;
  }

  summary::before {
    content: '▶';
    font-size: 0.75rem;
    transition: transform 0.2s;
  }

  details[open] summary::before {
    transform: rotate(90deg);
  }

  .content {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #4b5563;
  }
</style>
