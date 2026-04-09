# Be Warren Buffett

Someone on my feed was selling a "Buffett distillation" — they'd scraped his shareholder letters, stuffed them into a system prompt, and were charging $20/month for what was basically a chatbot in a costume. A commenter called it out: "That's not distillation. Distillation changes model weights. This is cosplay."

The commenter was technically right. And yet.

I tried it. I typed "Be Warren Buffett. Analyze this company for me." And the model started talking about moats. About intrinsic value. About margin of safety and long-term compounding. It didn't sound like a chatbot wearing a Buffett mask. It sounded like someone who had actually internalized a specific way of thinking about investing.

So I got curious. Not about whether the output was good advice — it probably wasn't — but about what happens inside the model when you type "Warren Buffett." What does that name actually do?

---

We opened up Gemma-2-2B, a small open-source model, using Anthropic's circuit tracing tools. These let you trace, at the level of individual computational features, what a token activates downstream. Not what the model says — what it computes.

Here's what we found. We gave the model "The investment philosophy of Warren Buffett is about ___" and let it tell us what comes next. We didn't define a vocabulary. We just read off the model's own top predictions and traced how much of each prediction came from the "Buffett" features specifically.

Buffett activated: *value*, *patience*, *dividends*, *compounding*, *quality*. Suppressed: *algorithmic*, *arbitrage*, *correlation*. The name didn't just nudge the output. It carved a shape — amplifying one region of concept space while dampening others.

We did the same for George Soros. Different shape entirely. *Crisis*, *emerging*, *geopolitical*, *reflexivity* — all elevated. *Moat* and *intrinsic value* — suppressed. The two investors' names are almost mirror images: what one activates, the other pushes down.

Jim Simons: *statistical* went up 1,471x over baseline. *Algorithmic*, *quantitative*, *systematic*. The quant's name activates quant concepts. Not because we told the model this. Because that's where the pointer points.

And "a random person"? Flat. Generic. *Diversification*, *risk*, *stocks* — the default answers you'd give if you didn't know anything in particular. No shape. No signature. The token points nowhere.

Then the causal test. We corrupted "Buffett" — replaced it with noise while keeping the rest of the prompt intact. The model's probability of saying "Berkshire" dropped from 70.6% to 0.0%. Destroy the pointer, the knowledge vanishes. For "a random person is the CEO of ___", corrupting the name changed nothing. There was never any knowledge to lose.

---

Wait — this isn't just about investors.

We ran the same experiment on philosophers. Gave the model "The philosophy of Karl Marx is fundamentally about ___" and traced what the name activated. Again, no predefined concept list. The model discovered its own associations.

Marx: *revolution*, *capitalism*, *inequality*, *material*, *economic*, *class*. Kant: *moral*, *ethics*, *rationality*, *critical*. Sartre: *existential*, *freedom*, *questioning*. We never said these words. The names surfaced them.

This is the thing: a proper noun in a language model isn't a label. It's an address. Two tokens — "Karl" and "Marx" — compress an entire intellectual tradition into a location in the model's computational space. When those tokens enter the residual stream, they activate a structured manifold of associated concepts. Not a fact. Not a definition. A geometry.

---

I realize this might sound like a curiosity about language models. It's not. It's a property of language itself.

Every proper noun works this way. "Einstein" doesn't mean "a physicist." It means *that* physicist — the one who bent spacetime, who wrote to Roosevelt, who said God doesn't play dice. The name is a compressed address into a web of associations. When you hear it, your brain doesn't look up a dictionary entry. It activates a shape.

This is the weak Sapir-Whorf hypothesis, and it's more defensible than people give it credit for. Language doesn't just describe thought. It provides handles. The word "justice" isn't a description of a concept — it's a grip that lets you pick the concept up, rotate it, compare it to "fairness," argue about where it breaks down. Without the handle, the concept is still there, but it's harder to hold and impossible to pass to someone else.

A lion hunts with intelligence. It reads terrain, predicts prey movement, coordinates with the pride. But that intelligence stays in the lion. It can't be copied, accumulated, or debugged. Language is what makes intelligence portable. "Flank left, I'll drive it toward the river" compresses a spatial plan into a sentence another mind can execute. Over centuries, these compressions stack: tactics become strategy, strategy becomes doctrine, doctrine becomes a name. "Sun Tzu."

---

That's what struck me most in the data. When we tested five philosophers, three out of five activated their own idea cluster as the single strongest signal. The model learned from the statistics of human text that "Marx" points to *dialectical* and *proletariat*, that "Kant" points to *transcendental* and *categorical*. The names became coordinates.

This is a property of culture, not just of neural networks. What does it mean to have lived a meaningful intellectual life? Maybe this: your name becomes a reliable address. "Darwinian" means something. "Keynesian" means something. "Machiavellian" means something — even to people who haven't read *The Prince*. The pointer propagates through culture independent of the source text. You don't need to read *The Wealth of Nations* to use "invisible hand." The address has been dereferenced so many times by so many minds that the ideas now live in the coordinate system itself.

Most of us won't have our names become adjectives. But the mechanism is the same at every scale. A teacher whose student says "she taught me to think critically" — that teacher has become a pointer in that student's mind. A parent whose kid says "my dad always said check twice" — a compressed heuristic, a token in someone else's reasoning.

Inside Gemma-2-2B, "Karl Marx" is literally a direction in a 2,304-dimensional space. When it enters the computation, it causally activates *revolution* and *class* and suppresses other directions. The life's work became a geometry. The philosopher became a vector.

Maybe that's what a life is for. To become a pointer that, when dereferenced, activates something worth thinking about.

---

*Code and data for all experiments: [github.com/xuy/lang-tokens](https://github.com/xuy/lang-tokens). Conducted on Gemma-2-2B using Anthropic's open-source circuit tracing tools.*
