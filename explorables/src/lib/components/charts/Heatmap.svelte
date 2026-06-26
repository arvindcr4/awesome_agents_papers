<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  export let data: number[][] = [];
  export let width = 400;
  export let height = 400;
  export let xLabels: string[] = [];
  export let yLabels: string[] = [];
  export let title = '';
  export let colorScheme: 'viridis' | 'blues' | 'reds' | 'diverging' = 'viridis';

  let container: HTMLDivElement;
  let tooltipText = '';
  let tooltipVisible = false;
  let tooltipX = 0;
  let tooltipY = 0;

  const margin = { top: 50, right: 30, bottom: 50, left: 60 };

  onMount(() => {
    createHeatmap();
  });

  function showTooltip(event: MouseEvent, yLabel: string, xLabel: string, value: number) {
    tooltipText = `${yLabel}, ${xLabel}: ${value.toFixed(3)}`;
    tooltipVisible = true;
    tooltipX = event.clientX + 10;
    tooltipY = event.clientY - 10;
  }

  function hideTooltip() {
    tooltipVisible = false;
  }

  function createHeatmap() {
    d3.select(container).selectAll('svg').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const flatData = data.flat();
    const [minVal, maxVal] = d3.extent(flatData) as [number, number];

    let colorScale: d3.ScaleSequential<string>;
    switch (colorScheme) {
      case 'blues':
        colorScale = d3.scaleSequential(d3.interpolateBlues).domain([minVal, maxVal]);
        break;
      case 'reds':
        colorScale = d3.scaleSequential(d3.interpolateReds).domain([minVal, maxVal]);
        break;
      case 'diverging':
        colorScale = d3.scaleSequential(d3.interpolateRdBu).domain([maxVal, minVal]);
        break;
      default:
        colorScale = d3.scaleSequential(d3.interpolateViridis).domain([minVal, maxVal]);
    }

    const rows = data.length;
    const cols = data[0]?.length || 0;

    const cellWidth = innerWidth / cols;
    const cellHeight = innerHeight / rows;

    if (title) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-weight', '500')
        .attr('font-size', '14px')
        .text(title);
    }

    data.forEach((row, i) => {
      row.forEach((value, j) => {
        const xLabel = xLabels[j] || j.toString();
        const yLabel = yLabels[i] || i.toString();

        g.append('rect')
          .attr('x', j * cellWidth)
          .attr('y', i * cellHeight)
          .attr('width', cellWidth - 1)
          .attr('height', cellHeight - 1)
          .attr('fill', colorScale(value))
          .attr('rx', 2)
          .style('cursor', 'pointer')
          .on('mouseenter', function(event) {
            showTooltip(event, yLabel, xLabel, value);
            d3.select(this)
              .attr('stroke', '#333')
              .attr('stroke-width', 2);
          })
          .on('mousemove', function(event) {
            tooltipX = event.clientX + 10;
            tooltipY = event.clientY - 10;
          })
          .on('mouseleave', function() {
            hideTooltip();
            d3.select(this).attr('stroke', 'none');
          });
      });
    });

    // Axis labels
    if (xLabels.length) {
      const step = Math.ceil(cols / 10);
      xLabels.forEach((label, i) => {
        if (i % step === 0) {
          g.append('text')
            .attr('x', i * cellWidth + cellWidth / 2)
            .attr('y', innerHeight + 15)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#666')
            .text(label);
        }
      });
    }

    if (yLabels.length) {
      const step = Math.ceil(rows / 10);
      yLabels.forEach((label, i) => {
        if (i % step === 0) {
          g.append('text')
            .attr('x', -5)
            .attr('y', i * cellHeight + cellHeight / 2 + 4)
            .attr('text-anchor', 'end')
            .attr('font-size', '10px')
            .attr('fill', '#666')
            .text(label);
        }
      });
    }

    // Color legend
    const legendWidth = 15;
    const legendHeight = innerHeight;
    const legendScale = d3.scaleLinear().domain([maxVal, minVal]).range([0, legendHeight]);

    const legend = svg.append('g')
      .attr('transform', `translate(${width - 25}, ${margin.top})`);

    const gradientId = `gradient-${Math.random().toString(36).substr(2, 9)}`;
    const legendGradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '0%')
      .attr('y1', '0%')
      .attr('y2', '100%');

    legendGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', colorScale(maxVal));

    legendGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', colorScale(minVal));

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', `url(#${gradientId})`);

    legend.append('g')
      .attr('transform', `translate(${legendWidth}, 0)`)
      .call(d3.axisRight(legendScale).ticks(5).tickFormat(d3.format('.2f')))
      .attr('font-size', '9px');
  }

  $: if (data.length && container) {
    createHeatmap();
  }
</script>

<div class="heatmap-wrapper">
  <div bind:this={container} class="heatmap-container"></div>
  {#if tooltipVisible}
    <div class="tooltip" style="left: {tooltipX}px; top: {tooltipY}px;">
      {tooltipText}
    </div>
  {/if}
</div>

<style>
  .heatmap-wrapper {
    position: relative;
    display: inline-block;
  }

  .tooltip {
    position: fixed;
    padding: 8px 12px;
    background: rgba(0, 0, 0, 0.85);
    color: white;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    z-index: 1000;
    white-space: nowrap;
  }
</style>
