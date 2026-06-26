// Manim-style animation components
export { default as ManimScene } from './ManimScene.svelte';
export { default as AnimatedPath } from './AnimatedPath.svelte';
export { default as AnimatedCircle } from './AnimatedCircle.svelte';
export { default as AnimatedText } from './AnimatedText.svelte';
export { default as AnimatedArrow } from './AnimatedArrow.svelte';

// Manim color palette
export const COLORS = {
  BLUE: '#58C4DD',
  TEAL: '#5CD0B3',
  GREEN: '#83C167',
  YELLOW: '#FFFF00',
  GOLD: '#F0AC5F',
  RED: '#FC6255',
  MAROON: '#C55F73',
  PURPLE: '#9A72AC',
  WHITE: '#FFFFFF',
  GREY: '#888888',
  DARK_GREY: '#444444',
  BACKGROUND: '#1a1a2e'
};

// Easing presets matching Manim
export const EASING = {
  smooth: 'power2.inOut',
  linear: 'none',
  rushInto: 'power2.in',
  rushFrom: 'power2.out',
  slowInto: 'power4.in',
  doubleSmooth: 'power4.inOut',
  thereAndBack: 'power2.inOut',
  wiggle: 'elastic.out(1, 0.3)'
};
