// Distill.pub-style components
export { default as Footnote } from './Footnote.svelte';
export { default as MarginNote } from './MarginNote.svelte';
export { default as Collapsible } from './Collapsible.svelte';
export { default as ParameterSlider } from './ParameterSlider.svelte';
export { default as StepControls } from './StepControls.svelte';

// Layout class names (use in class attributes)
export const LAYOUT = {
  body: 'l-body',        // Standard text width
  page: 'l-page',        // Slightly wider
  middle: 'l-middle',    // Centered, medium width
  screen: 'l-screen',    // Full width
  gutter: 'l-gutter',    // Margin area
  outset: 'l-outset',    // Extends into margins
};
