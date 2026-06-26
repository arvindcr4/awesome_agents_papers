<script lang="ts">
  /**
   * Distill-style parameter slider control.
   * Ties to visualizations through bound value.
   */
  import { createEventDispatcher } from 'svelte';

  export let label: string;
  export let value: number;
  export let min: number = 0;
  export let max: number = 100;
  export let step: number = 1;
  export let unit: string = '';
  export let showValue: boolean = true;

  const dispatch = createEventDispatcher();

  function handleInput(event: Event) {
    const target = event.target as HTMLInputElement;
    value = parseFloat(target.value);
    dispatch('change', value);
  }
</script>

<div class="parameter-control">
  <label>
    <span class="label-text">{label}</span>
    <input
      type="range"
      {min}
      {max}
      {step}
      {value}
      on:input={handleInput}
    />
    {#if showValue}
      <span class="value">{value}{unit}</span>
    {/if}
  </label>
</div>

<style>
  .parameter-control {
    margin: 0.75rem 0;
    font-family: 'Inter', sans-serif;
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  .label-text {
    font-size: 0.9rem;
    color: #374151;
    min-width: 120px;
  }

  input[type="range"] {
    flex: 1;
    min-width: 150px;
    height: 6px;
    -webkit-appearance: none;
    appearance: none;
    background: #e5e7eb;
    border-radius: 3px;
    cursor: pointer;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    background: #3b82f6;
    border-radius: 50%;
    cursor: pointer;
    transition: transform 0.1s;
  }

  input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.1);
  }

  input[type="range"]::-moz-range-thumb {
    width: 18px;
    height: 18px;
    background: #3b82f6;
    border: none;
    border-radius: 50%;
    cursor: pointer;
  }

  .value {
    font-size: 0.9rem;
    font-weight: 500;
    color: #3b82f6;
    min-width: 50px;
    text-align: right;
    font-family: 'SF Mono', Monaco, monospace;
  }
</style>
