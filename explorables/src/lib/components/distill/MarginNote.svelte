<script lang="ts">
  /**
   * Distill-style margin note.
   * Appears in the right gutter on desktop, inline on mobile.
   */
  export let id: string = '';
</script>

<aside class="margin-note" {id}>
  <slot />
</aside>

<style>
  .margin-note {
    position: relative;
    font-size: 0.85rem;
    line-height: 1.5;
    color: #6b7280;
    padding-left: 1rem;
    border-left: 2px solid #e5e7eb;
    margin: 1rem 0;
  }

  @media (min-width: 1024px) {
    .margin-note {
      position: absolute;
      right: -200px;
      width: 180px;
      margin: 0;
      padding-left: 0.75rem;
    }
  }
</style>
