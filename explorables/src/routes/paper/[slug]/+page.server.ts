import type { PageServerLoad } from './$types';
import { error } from '@sveltejs/kit';
import fs from 'fs';
import path from 'path';

export const load: PageServerLoad = async ({ params }) => {
  const { slug } = params;

  const dataDir = path.join(process.cwd(), 'src/lib/data/papers');
  const filePath = path.join(dataDir, `${slug}.json`);

  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const paper = JSON.parse(content);
    return { paper };
  } catch {
    throw error(404, 'Paper not found');
  }
};
