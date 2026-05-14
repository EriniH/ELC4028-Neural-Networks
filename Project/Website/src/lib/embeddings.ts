export type WordVec = {
  word: string;
  group: keyof typeof GROUP_COLORS;
  x: number;
  y: number;
};

export const GROUP_COLORS = {
  royalty: "#fbbf24",
  gender: "#f472b6",
  country: "#60a5fa",
  capital: "#22d3ee",
  profession: "#34d399",
} as const;

export const WORDS: WordVec[] = [
  { word: "king", group: "royalty", x: 2.4, y: 2.1 },
  { word: "queen", group: "royalty", x: 2.1, y: 2.0 },
  { word: "prince", group: "royalty", x: 2.6, y: 1.8 },
  { word: "princess", group: "royalty", x: 2.2, y: 1.6 },
  { word: "man", group: "gender", x: 0.6, y: -0.2 },
  { word: "woman", group: "gender", x: 0.3, y: -0.4 },
  { word: "boy", group: "gender", x: 0.7, y: -0.5 },
  { word: "girl", group: "gender", x: 0.4, y: -0.7 },
  { word: "france", group: "country", x: -2.5, y: 1.2 },
  { word: "italy", group: "country", x: -2.2, y: 1.0 },
  { word: "egypt", group: "country", x: -2.7, y: 0.9 },
  { word: "paris", group: "capital", x: -2.6, y: 0.5 },
  { word: "rome", group: "capital", x: -2.3, y: 0.4 },
  { word: "cairo", group: "capital", x: -2.8, y: 0.3 },
  { word: "doctor", group: "profession", x: 1.0, y: 1.2 },
  { word: "nurse", group: "profession", x: 0.8, y: 1.0 },
];

export function similarity(a: WordVec, b: WordVec) {
  const dot = a.x * b.x + a.y * b.y;
  const mag = Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y);
  if (mag === 0) return 0;
  return dot / mag;
}
