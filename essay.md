# Be Warren Buffett

*Why language tokens are the most efficient compression of intelligence we have*

---

"Be Warren Buffett. Now analyze this investment."

This is a real prompt people give to ChatGPT. And it works — sort of. The model starts talking about moats, intrinsic value, long-term compounding, margin of safety. It sounds like Buffett. A cottage industry has emerged around this trick: stuff a system prompt with someone's books, call it "distillation," charge a subscription. One Chinese commenter put it bluntly: "I just saw someone claiming they 'successfully distilled Buffett.' Distillation changes model weights. Copying someone's books into a system prompt is just cosplay."

The commenter is right about the terminology. But the cosplay works better than it should. Why?

## The pointer, not the label

Here's what we found when we opened up the model's internals.

When a language model reads "Warren Buffett," those two tokens don't function as a label meaning "a male American investor." They function as a **pointer** — an address that dereferences into a structured manifold of associated concepts inside the model's representation space.

We tested this using Anthropic's circuit tracing tools on Gemma-2-2B, a 2-billion-parameter open model. The method: give the model "The investment philosophy of Warren Buffett is about ___" and trace, at the circuit level, how much the features activated by the name "Buffett" contribute to each concept the model predicts next. We didn't define a list of concepts — the model told us what it associates with each name.

The results:

When the model reads **"Karl Marx,"** it activates *revolution*, *capitalism*, *inequality*, *material*, *economic*, *class*. When it reads **"Immanuel Kant,"** it activates *moral*, *ethics*, *rationality*, *critical*, *rational*. When it reads **"Sartre,"** it activates *existential*, *freedom*, *questioning*. When it reads **"a random person,"** it activates... *understanding*, *truth*, *knowledge* — vague, generic, pointed nowhere.

We didn't tell the model any of this. We just read what the name activated.

The same pattern holds for investors. Buffett's name activates value analysis and long-term compounding concepts 404 times more strongly than the baseline. George Soros activates macro and geopolitical concepts — *crisis*, *emerging*, *reflexivity* — while suppressing *moat* and *intrinsic value*. Jim Simons activates *statistical* (1,471x above baseline), *algorithmic*, *quantitative*. Each name carves a distinct shape across the concept space. "A random person" produces no shape at all.

The causal test is even starker. When we corrupt the "Buffett" tokens with noise — destroying the pointer while keeping the rest of the prompt intact — the model's probability of predicting "Berkshire" drops from 70.6% to 0.0%. Destroy the name, destroy the knowledge. "A random person is the CEO of ___" never had any knowledge to lose.

This is what's happening when you prompt "Be Warren Buffett." You're not doing cosplay. You're handing the model a pointer — a compressed address that unpacks into a structured web of concepts, associations, suppressions, and tendencies. The two tokens "Warren Buffett" are a key that unlocks a specific region of the model's learned representation space. It's not that the model pretends to be Buffett. It's that those tokens activate a computational pathway that was shaped by every mention of Buffett in the training data — his books, interviews, shareholder letters, commentary about him — all compressed into the geometry of where those tokens point.

## Language as compressed addressing

This is not a quirk of language models. It's a feature of language itself.

Every proper noun you know works this way. "Einstein" doesn't mean "a physicist." It means *that specific physicist* — the one who reimagined spacetime, played violin, had messy hair, wrote letters to Roosevelt, said "God does not play dice." The word is a compressed pointer to a web of facts, associations, and implications. When you hear "Einstein," your brain doesn't retrieve a dictionary entry. It activates a structured manifold of meaning, much like the model does.

This is what the Sapir-Whorf hypothesis points at, in its weaker and more defensible form: language doesn't just describe thought, it provides the infrastructure for it. The word "justice" is not a description of a concept — it's a handle that lets you pick up the concept, rotate it, compare it to "fairness" and "equality," argue about where it applies. Without the handle, the concept is harder to manipulate, harder to share, harder to refine across generations.

Tokens, in this view, are the most efficient compression of intelligence we have. A proper name compresses an entire life's work into two syllables. A technical term compresses a century of research into a single word. "Entropy" points to Boltzmann, Shannon, Jaynes, thermodynamics, information theory, and the heat death of the universe — all accessible through one six-letter address. Language gives intelligence a coordinate system.

## The lion's strategy

Consider a lion hunting a gazelle. The lion reasons — it reads terrain, predicts the prey's movement, coordinates with the pride, adjusts in real time. This is intelligence. But without language, this intelligence is trapped in the lion. It cannot be shared precisely, accumulated across generations, debugged, or recombined with other strategies.

Language is what makes intelligence *portable.* A human hunter can say "flank left, I'll drive it toward the river" — compressing a spatial plan into a sentence that another mind can decode and execute. Over generations, these compressed plans accumulate into tactics, then strategy, then written doctrine. The compression is lossy, but the addressability makes it worth it. You can point to "Sun Tzu's strategy" and activate a coherent body of military thought in someone who has never seen a battlefield.

This is exactly what "Be Warren Buffett" does in a language model — and why it's more than cosplay. The token isn't invoking a costume. It's dereferencing an address.

## What a name is for

Here's a strange thing we noticed in the data. When we tested five philosophers, three out of five activated their own idea cluster as the single strongest signal: Kant → Kantian ideas, Marx → Marxist ideas, Nietzsche → Nietzschean ideas. The model has never read a philosophy syllabus that organizes the world this way. It learned, from the statistical structure of human text, that "Marx" points to *dialectical*, *proletariat*, *alienation*, *bourgeois* — and that "Kant" points to *transcendental*, *categorical*, *rational*, *imperative*. The names became addresses for the ideas.

This is not just a property of language models. It's a property of culture.

What does it mean to have lived a meaningful intellectual life? One answer: your name becomes a reliable pointer. "Darwinian" means something. "Keynesian" means something. "Machiavellian" means something — even to people who have never read *The Prince*. The name has become a compressed address for a coherent set of ideas, and that address propagates through culture independently of the original text. You don't need to read *The Wealth of Nations* to use the concept "invisible hand." The pointer has been dereferenced so many times by so many minds that the ideas it addresses have become part of the shared coordinate system.

Most of us won't have our names become adjectives. But the mechanism is the same at every scale. A teacher whose students say "she taught me to think critically" has become a pointer in those students' minds — an address that activates a way of approaching problems. A parent whose child says "my dad always said to check twice" has become a compressed heuristic, a token in someone else's reasoning. You don't need to be Kant. You just need to be a pointer that, when dereferenced, activates something coherent.

The philosophers whose names survived centuries did so not because they had the most original thoughts — many of their ideas were anticipated by predecessors — but because they compressed their thinking into a form that could be addressed by a single word. They gave their ideas a name: their own. And that name became a coordinate in the space of all ideas, a fixed point that other thinkers could navigate relative to. "I am a Kantian." "I reject the Marxist framework." "This is a Nietzschean move." The names are not descriptions. They are coordinates.

Language models make this visible in a way that's almost uncomfortably literal. Inside Gemma-2-2B, "Karl Marx" is a vector. That vector, when it enters the computation, causally activates *revolution* and *capitalism* and *class* and suppresses other concepts. The philosopher became a direction in a high-dimensional space. The life's work became a geometry.

Perhaps that's what the meaning of a life is: to become a pointer that, when dereferenced, activates something worth thinking.

---

*Technical details and reproducible code for all experiments described in this essay are available at [github.com/xuy/lang-tokens](https://github.com/xuy/lang-tokens). Experiments were conducted on Gemma-2-2B using Anthropic's open-source circuit tracing tools.*
