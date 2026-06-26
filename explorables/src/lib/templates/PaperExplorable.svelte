<script lang="ts">
  import LineChart from '$lib/components/charts/LineChart.svelte';
  import Heatmap from '$lib/components/charts/Heatmap.svelte';
  import { ManimScene, AnimatedCircle, AnimatedText, AnimatedArrow, AnimatedPath, COLORS } from '$lib/components/manim';
  import { Collapsible } from '$lib/components/distill';

  export let paper: {
    title: string;
    authors: string[];
    abstract: string;
    category: string;
    year: number;
    arxivId?: string;
    sections: Section[];
  };

  interface Section {
    title: string;
    content: string;
    visualization?: Visualization;
  }

  interface Visualization {
    type: 'line-chart' | 'heatmap' | 'neural-network' | 'flow-diagram' | 'custom';
    data?: any;
    config?: any;
  }

  const categoryColors: Record<string, string> = {
    'agent-frameworks': '#3b82f6',
    'benchmarks': '#10b981',
    'computer-use': '#f59e0b',
    'memory-rag': '#8b5cf6',
    'multi-agent': '#ec4899',
    'planning': '#06b6d4',
    'reasoning': '#6366f1',
    'rl-finetuning': '#f97316',
    'robotics': '#14b8a6',
    'security': '#ef4444',
    'theorem-proving': '#a855f7',
    'web-agents': '#84cc16'
  };

  const categoryDisplay: Record<string, string> = {
    'agent-frameworks': 'Agent Frameworks',
    'benchmarks': 'Benchmarks',
    'computer-use': 'Computer Use',
    'memory-rag': 'Memory & RAG',
    'multi-agent': 'Multi-Agent',
    'planning': 'Planning',
    'reasoning': 'Reasoning',
    'rl-finetuning': 'RL Fine-tuning',
    'robotics': 'Robotics',
    'security': 'Security',
    'theorem-proving': 'Theorem Proving',
    'web-agents': 'Web Agents'
  };

  function getCategoryColor(category: string): string {
    return categoryColors[category] || '#6b7280';
  }

  // Calculate flow diagram box positions
  function getFlowBoxX(idx: number, total: number, width: number): number {
    const spacing = width / (total + 1);
    return spacing * (idx + 1);
  }
</script>

<svelte:head>
  <title>{paper.title} | AI Agents Explorables</title>
  <meta name="description" content={paper.abstract?.slice(0, 160)}>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
</svelte:head>

<article class="paper-explorable">
  <header>
    <a href="/" class="back-link-top">← All Papers</a>
    <span class="category" style="background: {getCategoryColor(paper.category)}">
      {categoryDisplay[paper.category] || paper.category}
    </span>
    <h1>{paper.title}</h1>
    <p class="authors">{paper.authors?.join(', ')}</p>
    <p class="meta">
      {paper.year}
      {#if paper.arxivId}
        • <a href="https://arxiv.org/abs/{paper.arxivId}" target="_blank" rel="noopener">arXiv:{paper.arxivId}</a>
      {/if}
    </p>
  </header>

  <!-- Distill-style lead question -->
  <section class="lead-question">
    <div class="question-mark">?</div>
    <p>{paper.abstract}</p>
  </section>

  {#each paper.sections as section, i}
    <section class="content-section">
      <h2>
        <span class="section-number">{i + 1}</span>
        {section.title}
      </h2>
      <div class="section-content">
        <p>{@html section.content}</p>
      </div>

      {#if section.visualization}
        <figure class="visualization">
          {#if section.visualization.type === 'line-chart' && section.visualization.data?.series}
            <LineChart
              series={section.visualization.data.series}
              width={600}
              height={300}
              xLabel={section.visualization.config?.xLabel || 'Steps'}
              yLabel={section.visualization.config?.yLabel || 'Value'}
              title=""
              chartId={`chart-${i}`}
            />
          {:else if section.visualization.type === 'heatmap' && section.visualization.data?.matrix}
            <Heatmap
              data={section.visualization.data.matrix}
              width={400}
              height={400}
              title=""
              colorScheme={section.visualization.config?.colorScheme || 'viridis'}
            />
          {:else if section.visualization.type === 'neural-network' && section.visualization.data?.layers}
            <ManimScene width={700} height={350} autoplay={true}>
              {#each section.visualization.data.layers as layer, layerIdx}
                {#each layer.nodes as node, nodeIdx}
                  <AnimatedCircle
                    cx={120 + layerIdx * 230}
                    cy={80 + nodeIdx * 70}
                    r={28}
                    stroke={layer.color || [COLORS.BLUE, COLORS.TEAL, COLORS.GREEN][layerIdx % 3]}
                    delay={layerIdx * 0.3 + nodeIdx * 0.1}
                  />
                {/each}
              {/each}
              <!-- Connection lines -->
              {#each section.visualization.data.layers as layer, layerIdx}
                {#if layerIdx < section.visualization.data.layers.length - 1}
                  {#each layer.nodes as _, fromIdx}
                    {#each section.visualization.data.layers[layerIdx + 1].nodes as __, toIdx}
                      <AnimatedArrow
                        x1={148 + layerIdx * 230}
                        y1={80 + fromIdx * 70}
                        x2={92 + (layerIdx + 1) * 230}
                        y2={80 + toIdx * 70}
                        stroke={COLORS.GREY}
                        strokeWidth={1}
                        delay={1.5 + layerIdx * 0.2 + fromIdx * 0.05 + toIdx * 0.05}
                      />
                    {/each}
                  {/each}
                {/if}
              {/each}
              <!-- Layer labels -->
              <AnimatedText x={120} y={320} text="Input" fill={COLORS.WHITE} fontSize={13} delay={2.5} />
              <AnimatedText x={350} y={320} text="Hidden" fill={COLORS.WHITE} fontSize={13} delay={2.6} />
              <AnimatedText x={580} y={320} text="Output" fill={COLORS.WHITE} fontSize={13} delay={2.7} />
            </ManimScene>
          {:else if section.visualization.type === 'flow-diagram' && section.visualization.data?.steps}
            {@const steps = section.visualization.data.steps}
            <ManimScene width={800} height={200} autoplay={true}>
              {#each steps as step, idx}
                <!-- Box -->
                <AnimatedPath
                  d={`M ${getFlowBoxX(idx, steps.length, 800) - 50} 70
                      L ${getFlowBoxX(idx, steps.length, 800) + 50} 70
                      L ${getFlowBoxX(idx, steps.length, 800) + 50} 130
                      L ${getFlowBoxX(idx, steps.length, 800) - 50} 130 Z`}
                  stroke={COLORS.TEAL}
                  strokeWidth={2}
                  delay={idx * 0.4}
                  duration={0.3}
                />
                <AnimatedText
                  x={getFlowBoxX(idx, steps.length, 800)}
                  y={105}
                  text={step.label}
                  fill={COLORS.WHITE}
                  fontSize={13}
                  delay={idx * 0.4 + 0.2}
                />
                <!-- Arrow to next -->
                {#if idx < steps.length - 1}
                  <AnimatedArrow
                    x1={getFlowBoxX(idx, steps.length, 800) + 55}
                    y1={100}
                    x2={getFlowBoxX(idx + 1, steps.length, 800) - 55}
                    y2={100}
                    stroke={COLORS.GREY}
                    strokeWidth={2}
                    delay={idx * 0.4 + 0.3}
                  />
                {/if}
              {/each}
            </ManimScene>
          {/if}
          <figcaption>{section.visualization.config?.title || section.title}</figcaption>
        </figure>
      {/if}
    </section>
  {/each}

  <!-- Key takeaways -->
  <section class="takeaways">
    <h2>Key Takeaways</h2>
    <ul>
      {#each paper.sections.slice(0, 3) as section}
        <li><strong>{section.title}:</strong> {section.content?.replace(/<[^>]*>/g, '').slice(0, 100)}...</li>
      {/each}
    </ul>
  </section>

  <footer>
    <div class="footer-links">
      <a href="/" class="back-link">← All Papers</a>
      {#if paper.arxivId}
        <a href="https://arxiv.org/pdf/{paper.arxivId}" target="_blank" rel="noopener" class="pdf-link">
          📄 Read Full Paper
        </a>
      {/if}
    </div>
    <p class="attribution">
      Interactive explorable generated from research paper.
      Original work by {paper.authors?.slice(0, 2).join(', ')}{paper.authors?.length > 2 ? ' et al.' : ''} ({paper.year}).
    </p>
  </footer>
</article>

<style>
  .paper-explorable {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
    font-family: 'Source Serif 4', Georgia, serif;
  }

  .back-link-top {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #6b7280;
    text-decoration: none;
    margin-bottom: 1.5rem;
  }

  .back-link-top:hover {
    color: #3b82f6;
  }

  header {
    margin-bottom: 2rem;
  }

  .category {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    color: white;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
  }

  h1 {
    font-size: 2.25rem;
    font-weight: 600;
    line-height: 1.25;
    margin-bottom: 0.75rem;
    color: #111;
  }

  .authors {
    font-size: 1.05rem;
    color: #4b5563;
    margin-bottom: 0.5rem;
  }

  .meta {
    font-size: 0.9rem;
    color: #6b7280;
    font-family: 'Inter', sans-serif;
  }

  .meta a {
    color: #3b82f6;
    text-decoration: none;
  }

  .meta a:hover {
    text-decoration: underline;
  }

  /* Distill-style lead question */
  .lead-question {
    position: relative;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    padding: 2rem 2rem 2rem 4rem;
    border-radius: 12px;
    margin-bottom: 3rem;
    border-left: 4px solid #3b82f6;
  }

  .question-mark {
    position: absolute;
    left: 1rem;
    top: 1.5rem;
    font-size: 2rem;
    font-weight: 700;
    color: #3b82f6;
    opacity: 0.3;
  }

  .lead-question p {
    font-size: 1.15rem;
    line-height: 1.7;
    color: #374151;
    margin: 0;
  }

  h2 {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #1f2937;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .section-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: #f3f4f6;
    border-radius: 50%;
    font-size: 0.85rem;
    font-weight: 600;
    color: #6b7280;
    font-family: 'Inter', sans-serif;
  }

  .content-section {
    margin-bottom: 3rem;
  }

  .section-content p {
    font-size: 1.1rem;
    line-height: 1.8;
    color: #374151;
    margin-bottom: 1.5rem;
  }

  figure.visualization {
    margin: 2rem 0;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  figcaption {
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #6b7280;
    font-style: italic;
    text-align: center;
  }

  .takeaways {
    background: #fefce8;
    padding: 1.5rem 2rem;
    border-radius: 8px;
    margin: 3rem 0;
    border: 1px solid #fef08a;
  }

  .takeaways h2 {
    color: #854d0e;
    font-size: 1.2rem;
    margin-bottom: 1rem;
  }

  .takeaways ul {
    margin: 0;
    padding-left: 1.25rem;
  }

  .takeaways li {
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #713f12;
  }

  .takeaways strong {
    color: #854d0e;
  }

  footer {
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid #e5e7eb;
  }

  .footer-links {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .back-link, .pdf-link {
    font-family: 'Inter', sans-serif;
    text-decoration: none;
    font-size: 0.9rem;
  }

  .back-link {
    color: #6b7280;
  }

  .back-link:hover {
    color: #3b82f6;
  }

  .pdf-link {
    color: #3b82f6;
    padding: 0.5rem 1rem;
    background: #eff6ff;
    border-radius: 6px;
  }

  .pdf-link:hover {
    background: #dbeafe;
  }

  .attribution {
    font-size: 0.85rem;
    color: #9ca3af;
    text-align: center;
  }

  :global(.paper-explorable strong) {
    font-weight: 600;
    color: #111;
  }

  :global(.paper-explorable code) {
    background: #f3f4f6;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    font-size: 0.9em;
    font-family: 'SF Mono', Monaco, monospace;
  }

  @media (max-width: 768px) {
    h1 {
      font-size: 1.75rem;
    }

    .lead-question {
      padding: 1.5rem 1.5rem 1.5rem 3rem;
    }

    .footer-links {
      flex-direction: column;
      gap: 1rem;
      align-items: flex-start;
    }
  }
</style>
