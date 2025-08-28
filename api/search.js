import OpenAI from "openai";

// OpenAI API key z env proměnné
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// malý dataset přímo v kódu
const documents = [
  { title: "BMW M5", content: "BMW M5 je luxusní sportovní sedan s vysokým výkonem a elegantním designem." },
  { title: "Mercedes G-Class", content: "Mercedes G-Class je ikonické SUV známé svou robustností a luxusem." },
  { title: "Python programování", content: "Python je univerzální programovací jazyk vhodný pro web, AI a skriptování." },
  { title: "Node.js", content: "Node.js je runtime prostředí pro JavaScript, ideální pro backend aplikace." },
  { title: "OpenAI GPT", content: "OpenAI GPT je model umělé inteligence schopný generovat text a odpovídat na dotazy." }
];

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { query } = req.body;
  if (!query) return res.status(400).json({ error: "Missing query" });

  // 1) Vytvoření embedding pro dotaz
  const embeddingResp = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: query
  });
  const queryVector = embeddingResp.data[0].embedding;

  // 2) Jednoduché porovnání s datasetem přes cosine similarity
  function cosineSim(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  // embeddingy dokumentů
  const docEmbeddings = await Promise.all(documents.map(async doc => {
    const resp = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: doc.content
    });
    return { ...doc, embedding: resp.data[0].embedding };
  }));

  // top 3 nejbližší dokumenty
  const topDocs = docEmbeddings
    .map(doc => ({ ...doc, score: cosineSim(queryVector, doc.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  // 3) GPT odpověď
  const prompt = `Na základě těchto textů odpověz na dotaz: "${query}"\n\n` +
                 topDocs.map(d => `- ${d.title}: ${d.content}`).join("\n");

  const gptResp = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [{ role: "user", content: prompt }],
    temperature: 0.5
  });

  const answer = gptResp.choices[0].message.content;
  res.status(200).json({ answer, topDocs });
}
