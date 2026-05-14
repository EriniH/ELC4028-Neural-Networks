import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

type Lang = "en" | "ar";

type LanguageContextValue = {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: string) => string;
};

const translations: Record<Lang, Record<string, string>> = {
  en: {
    "nav.brand": "Word Embeddings",
    "nav.learn": "Learn",
    "nav.demo": "Demo",
    "nav.arabic": "Arabic",
    "nav.quiz": "Quiz",
    "nav.lang": "عربي",
    "hero.badge": "Neural Networks • Language",
    "hero.title1": "Understand word",
    "hero.title2": "embeddings",
    "hero.desc":
      "Learn what word embeddings are with clear explanations, an interactive 2D demo, and a quiz to test your understanding.",
    "hero.start": "Start learning",
    "hero.skip": "Skip to demo",
    "learn.eyebrow": "Concepts",
    "learn.title": "What are word embeddings?",
    "learn.p1":
      "Word embeddings map words to vectors so similar meanings end up close together in space.",
    "learn.p2":
      "These vectors let models compare words with cosine similarity and perform simple arithmetic to reveal relationships.",
    "learn.card1.t": "Similarity",
    "learn.card1.d": "Nearby vectors capture related meanings like king and queen.",
    "learn.card2.t": "Analogy",
    "learn.card2.d": "Vector math can model relations such as gender or geography.",
    "learn.card3.t": "Context",
    "learn.card3.d": "Modern models refine embeddings based on the surrounding words.",
    "learn.example": "Vector arithmetic",
    "learn.example.note":
      "Subtracting and adding vectors can reveal semantic relationships in the embedding space.",
    "demo.eyebrow": "Explore",
    "demo.title": "Interactive 2D demo",
    "demo.desc": "Click a word to see its nearest neighbors in the embedding space.",
    "demo.hint": "Try selecting different words to compare similarity patterns.",
    "demo.selected": "Selected word",
    "demo.neighbors": "Closest neighbors",
    "arabic.eyebrow": "Arabic",
    "arabic.title": "Arabic embeddings demo",
    "arabic.desc": "Explore an Arabic Word2Vec demo with an interactive 3D visualization.",
    "arabic.open": "Open the Arabic demo",
    "quiz.eyebrow": "Quiz",
    "quiz.title": "Quick check",
    "quiz.q": "Question",
    "quiz.of": "of",
    "quiz.score": "Score",
    "quiz.perfect": "Perfect score! You nailed it.",
    "quiz.nice": "Nice work! You’re getting it.",
    "quiz.again": "Give it another try to improve.",
    "quiz.try": "Try again",
    "quiz.results": "See results",
    "quiz.next": "Next question",
    footer: "Built for ELC4028 Neural Networks.",
  },
  ar: {
    "nav.brand": "تضمين الكلمات",
    "nav.learn": "تعلّم",
    "nav.demo": "العرض",
    "nav.arabic": "عربي",
    "nav.quiz": "اختبار",
    "nav.lang": "English",
    "hero.badge": "الشبكات العصبية • اللغة",
    "hero.title1": "افهم",
    "hero.title2": "تضمين الكلمات",
    "hero.desc": "تعلّم ما هي تضمينات الكلمات مع شرح واضح، عرض تفاعلي ثنائي الأبعاد، واختبار قصير.",
    "hero.start": "ابدأ التعلّم",
    "hero.skip": "اذهب للعرض",
    "learn.eyebrow": "المفاهيم",
    "learn.title": "ما هي تضمينات الكلمات؟",
    "learn.p1":
      "تحوّل تضمينات الكلمات الكلمات إلى متجهات بحيث تكون المعاني المتقاربة متجاورة في الفضاء.",
    "learn.p2": "تسمح هذه المتجهات بقياس التشابه وعمليات حسابية بسيطة لاكتشاف العلاقات.",
    "learn.card1.t": "التشابه",
    "learn.card1.d": "المتجهات المتقاربة تعبّر عن معانٍ مرتبطة.",
    "learn.card2.t": "القياس",
    "learn.card2.d": "عمليات المتجهات تلتقط علاقات مثل النوع أو الجغرافيا.",
    "learn.card3.t": "السياق",
    "learn.card3.d": "نماذج أحدث تضبط التضمين حسب السياق المحيط.",
    "learn.example": "حساب المتجهات",
    "learn.example.note": "الطرح والجمع بين المتجهات يمكن أن يكشف علاقات دلالية في الفضاء.",
    "demo.eyebrow": "استكشف",
    "demo.title": "عرض تفاعلي ثنائي الأبعاد",
    "demo.desc": "اضغط على كلمة لرؤية أقرب الجيران في الفضاء المتجهي.",
    "demo.hint": "جرّب كلمات مختلفة لمقارنة أنماط التشابه.",
    "demo.selected": "الكلمة المحددة",
    "demo.neighbors": "أقرب الكلمات",
    "arabic.eyebrow": "العربية",
    "arabic.title": "عرض التضمين العربي",
    "arabic.desc": "استكشف عرض Word2Vec العربي مع تصور ثلاثي الأبعاد.",
    "arabic.open": "افتح العرض العربي",
    "quiz.eyebrow": "اختبار",
    "quiz.title": "اختبار سريع",
    "quiz.q": "سؤال",
    "quiz.of": "من",
    "quiz.score": "النتيجة",
    "quiz.perfect": "درجة كاملة! رائع جدًا.",
    "quiz.nice": "عمل ممتاز! استمر.",
    "quiz.again": "جرّب مرة أخرى للتحسن.",
    "quiz.try": "أعد المحاولة",
    "quiz.results": "عرض النتائج",
    "quiz.next": "السؤال التالي",
    footer: "تم البناء لمقرر الشبكات العصبية ELC4028.",
  },
};

type QuizQuestion = {
  q: string;
  options: string[];
  answer: number;
  explain: string;
};

export const QUIZ_QUESTIONS: Record<Lang, QuizQuestion[]> = {
  en: [
    {
      q: "What does cosine similarity measure between two word vectors?",
      options: [
        "Their distance in meters",
        "Their angle in vector space",
        "Their length",
        "Their frequency",
      ],
      answer: 1,
      explain:
        "Cosine similarity compares the angle between vectors, capturing semantic similarity.",
    },
    {
      q: "Which operation helps model analogies like king − man + woman?",
      options: [
        "Vector addition/subtraction",
        "Sorting alphabetically",
        "Counting characters",
        "Stemming",
      ],
      answer: 0,
      explain: "Vector arithmetic reveals relational patterns in embedding space.",
    },
    {
      q: "Why are embeddings useful?",
      options: [
        "They keep words as plain text only",
        "They allow numeric comparison of meaning",
        "They guarantee perfect translations",
        "They remove all ambiguity",
      ],
      answer: 1,
      explain: "Embeddings turn words into vectors so models can compare meanings numerically.",
    },
  ],
  ar: [
    {
      q: "ما الذي يقيسه التشابه الكوسيني بين متجهين؟",
      options: ["المسافة بالأمتار", "الزاوية في الفضاء المتجهي", "الطول", "التكرار"],
      answer: 1,
      explain: "تشابه جيب التمام يقارن الزاوية بين المتجهات لقياس التشابه الدلالي.",
    },
    {
      q: "أي عملية تساعد في تمثيل القياسات مثل ملك − رجل + امرأة؟",
      options: ["جمع/طرح المتجهات", "الترتيب الأبجدي", "عد الحروف", "التجذير"],
      answer: 0,
      explain: "حساب المتجهات يكشف العلاقات في فضاء التضمين.",
    },
    {
      q: "لماذا تعد التضمينات مفيدة؟",
      options: [
        "لأنها تبقي الكلمات كنص فقط",
        "لأنها تسمح بمقارنة المعاني رقمياً",
        "لأنها تضمن ترجمة مثالية",
        "لأنها تزيل كل الغموض",
      ],
      answer: 1,
      explain: "تحول التضمينات الكلمات إلى متجهات يمكن مقارنتها رقمياً.",
    },
  ],
};

const LanguageContext = createContext<LanguageContextValue | null>(null);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>("en");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = window.localStorage.getItem("lang");
    if (saved === "en" || saved === "ar") {
      setLangState(saved);
    }
  }, []);

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("lang", next);
    }
  }, []);

  const t = useCallback(
    (key: string) => translations[lang]?.[key] ?? translations.en[key] ?? key,
    [lang],
  );

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLang() {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLang must be used within LanguageProvider");
  }
  return ctx;
}
