<script lang="ts">
  /**
   * Distill-style hoverable footnote.
   * Usage: <Footnote>Side remark here</Footnote>
   */
  export let number: number | null = null;

  let visible = false;
  let x = 0;
  let y = 0;

  function show(event: MouseEvent) {
    visible = true;
    x = event.clientX;
    y = event.clientY;
  }

  function hide() {
    visible = false;
  }

  function updatePosition(event: MouseEvent) {
    x = event.clientX;
    y = event.clientY;
  }
</script>

<span class="footnote-marker" on:mouseenter={show} on:mouseleave={hide} on:mousemove={updatePosition}>
  <sup>{number ?? '†'}</sup>
</span>

{#if visible}
  <div class="footnote-popup" style="left: {x + 10}px; top: {y + 10}px;">
    <slot />
  </div>
{/if}

<style>
  .footnote-marker {
    cursor: help;
    color: #3b82f6;
  }

  .footnote-marker sup {
    font-size: 0.75em;
    font-weight: 500;
  }

  .footnote-popup {
    position: fixed;
    max-width: 300px;
    padding: 0.75rem 1rem;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-size: 0.9rem;
    line-height: 1.5;
    color: #4b5563;
    z-index: 1000;
    pointer-events: none;
  }
</style>
