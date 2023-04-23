# InstructionZoo

A collection of open-source Instruction-tuning dataset to train chat-based LLMs (ChatGPT,LLaMA,Alpaca).

This is an on-going project. The format and explaination of the following contents will be updated soon. (By Zhihan)

# Table of Contents

# The template

```
## [owner/project-name](https://github.com/link/to/project)

* Size:
* Language:
* Summary:
* Generation Method:
* Paper:
* HuggingFace: (if applicable)
* Demo: (if applicable)
* License:
```

# The English Instruction Datasets

## [tatsu-lab/Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 
* Size: 175 seed instructions, 52,002 instructions
* Language: EN
* Summary: Alpaca contains 52K instruction-following data, consisting of instruction, input and output. 
* Generateion Method: Self-instruct with human written 175 seed tasks.
* Paper: [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
* HuggingFace: https://huggingface.co/datasets/tatsu-lab/alpaca
* License: [CC BY NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en_GB)

## [gururise/Cleaned Alpaca](https://github.com/gururise/AlpacaDataCleaned) 
* Size: 51,713 instructions
* Language: EN
* Summary: Cleaned Alpaca Dataset helps solve the folowing issues: Hallucinations, Merged Instructions, Empty outputs, Empty code examples, Instructions to generate images, N/A outputs, Inconsistent input field, Wrong answers, Non-Sensical/Unclear instructions, and Extraneous escape and control characters.
* HuggingFace: https://huggingface.co/datasets/yahma/alpaca-cleaned
* License: [CC BY NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en_GB)

## [PhoebusSi/Alpaca-COT](https://github.com/PhoebusSi/alpaca-CoT) 
* Language: EN
* Summary: Alpaca-COT is a datset for Chain-of-Thoughts reasoning based on LLaMA and Alpaca.
* Generateion Method: Use the template provided by FLAN to change the original dataset into various Chain-of-Thoughts forms, and then convert them to the instruction-input-output triplets.
* HuggingFace: https://huggingface.co/datasets/QingyiSi/Alpaca-CoT
* License: [Apache License](https://www.apache.org/licenses/LICENSE-2.0)

## [orhonovich/unnatural-instructions](https://github.com/orhonovich/unnatural-instructions) 
* Size: 240,000 instructions
* Language: EN
* Summary: Unnatural Instructions consist of a core dataset of 68,478 instruction-input-output triplets, and a full dataset.
* Generateion Method: 
  * Step 1 (Core Dataset Generation): Collect 64,000 examples by prompting a language model with three seed examples of instructions and eliciting a fourth, following a strict instruction-input-output format.
  * Step 2 (Template Expansion): Prompt a language model to reformulate the tasks in the core dataset, and collect two alternative formulations for each generated task
* Paper: [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/pdf/2212.09689.pdf)
* License:

## [bigscience/PromptSource](https://github.com/bigscience-workshop/promptsource) 
* Size: 180 tasks, 2,085 instructions
* Language: EN
* Summary: PromptSource aims at designing a prompt query such that the answer can be mapped onto the specific dataset
* Generateion Method:
  *  Five steps: Dataset Exploration, Prompt Writing, Prompt Documentation, Iteration and Variation, and Global Review.
* Paper: [PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts](https://arxiv.org/pdf/2202.01279.pdf)
* HuggingFace: https://huggingface.co/spaces/bigscience/promptsource/tree/main
* Demo: https://huggingface.co/spaces/bigscience/promptsource
* License:

## [bigscience/P3](https://github.com/bigscience-workshop/promptsource) 
* Size: 270 tasks, 2,085 instructions
* Language: EN 
* Summary: P3 has a diverse set of NLP tasks, including multiple-choice QA, sentiment analysis or natural language inference. 
* Generateion Method: A subset of the prompts available in Promptsource.
* Paper: [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/pdf/2110.08207.pdf)
* HuggingFace: https://huggingface.co/datasets/bigscience/P3
* License:

## [allenai/natural-instructions](https://github.com/allenai/natural-instructions) 
* Size: 61 tasks, 61 instructions
* Language: EN
* Summary: Natural Instruct v1 is a dataset of 61 distinct tasks, their human-authored instructions, and 193k task instances. 
* Generateion Method:
  * Map exist datasets into Instruction Schema.
  * Instruction Schema:
    * Part I - Title + Definition + Things-to-Avoid + Emphasis-and-Caution
    * Part II - Positive Example: Input + Output + Reason
    * Part III - Negative Example: Input + Output + Reason + Suggestions to be modified to be positive
    * Part IV - Prompt
* Paper: [Cross-Task Generalization via Natural Language Crowdsourcing Instructions](https://arxiv.org/pdf/2104.08773.pdf)
* Demo: https://instructions.apps.allenai.org/
* License:

## [allenai/super-natural-instructions](https://github.com/allenai/natural-instructions) 
* Size: 1,616 tasks, 1,616 instructions
* Language: EN
* Summary: Super-Natural-Instruct v2 is built on Natural Instruct v1, has a simpler schema and contains over 1.5k tasks.
* Generateion Method:
  * Map exist datasets into Instruction Schema.
  * Instruction Schema:
    * Part I - Definition
    * Part II - Positive Example: Input + Output + Reason
    * Part III - Negative Example: Input + Output + Reason + Suggestions to be modified to be positive
* Paper: [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/pdf/2204.07705.pdf)
* Demo: https://instructions.apps.allenai.org/
* License:

## [google-research/FLAN 2021](https://github.com/google-research/flan) 
* Size: 62 tasks
* Language: EN
* Summary: FLAN 2021 aggregates 62 text datasets on Tensorflow Datasets into a single mixture. It is currently not public.
* Generateion Method: Map exist datasets into Instruction Schema.
* Paper: [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/pdf/2109.01652.pdf)
* License:

## [google-research/FLAN 2022 Collection](https://github.com/google-research/FLAN/tree/main/flan/v2) 
* Size: 1,836 tasks, 18,360 instructions
* Language: EN
* Summary:  Flan 2022 Collection combines Flan 2021, P3 Dataset Family, Super-Natural Instructions, with some additional reasoning, dialog, and program synthesis datasets.
* Paper: [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/pdf/2301.13688.pdf)
* License:

## [LianjiaTech/BELLE 1.5M](https://github.com/LianjiaTech/BELLE/tree/main/1.5M) 
* Size: 175 seed instructions, 1.5M instructions
* Language: CH
* Summary: 1.5M Chinese instructions produced by BELLE, with various instruction types and domains.
* Generateion Method: Self-instruct with 175 Chinese seed tasks translated from the seed tasks in Alpaca, using text-davinci-003.
* Paper: [Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases](https://arxiv.org/abs/2303.14742)
* HuggingFace: 
  * 0.5M https://huggingface.co/datasets/BelleGroup/train_0.5M_CN
  * 1.0M https://huggingface.co/datasets/BelleGroup/train_1M_CN
* Demo: https://github.com/LianjiaTech/BELLE/blob/main/chat/README.md
* License: https://github.com/LianjiaTech/BELLE/blob/main/DISCLAIMER

## [LianjiaTech/BELLE 10M](https://github.com/LianjiaTech/BELLE/tree/main/10M) 
* Size: 10M instructions
* Language: CH
* Summary: 10M Chinese instructions produced by BELLE, with 4 subsets.
* Generateion Method:
  * School Math: Chinese math questions and answers generated by ChatGPT. 
  * Multiturn Chat: Chinese multiturn chat generated by ChatGPT, with two characters Human and Assistant.
  * Generated Chat: Chinese role-playing chat generated by ChatGPT.
  * 2M Chinese instructions: Various Chinese instructions generated by ChatGPT.
* Paper: [Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases](https://arxiv.org/abs/2303.14742)
* HuggingFace: 
  * School Math https://huggingface.co/datasets/BelleGroup/school_math_0.25M
  * Multiturn Chat https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M
  * Generated Chat https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M
  * 2M Chinese instructions https://huggingface.co/datasets/BelleGroup/train_2M_CN
* Demo: https://github.com/LianjiaTech/BELLE/blob/main/chat/README.md
* License: https://github.com/LianjiaTech/BELLE/blob/main/DISCLAIMER

## [XueFuzhao/InstructionWild](https://github.com/XueFuzhao/InstructionWild)
* Size: 479 seed instructions, 52,191 Chinese instructions, 52,191 English instructions
* Language: CH, EN
* Summary: InstructionWild use the same format as Alpaca for fast and easy usage. Its instructions have no input field.
* Generateion Method: 
  * Pick 429 instructions over 700 noisy instructions from Twitter
  * Use a similar method as Alpaca for generating the resulting instructions.
* License:

## ExMix

* Paper: [ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning](https://arxiv.org/pdf/2111.10952.pdf)
* Download: ExMix's official data is not open-sourced, but you can use the following URLs to download partial data in ExMiX.
  * [COGS](https://github.com/najoungkim/COGS)
  * [Shakespearizing-Modern-English](https://github.com/harsh19/)
  * [Shakespearizing-Modern-English StylePTB](https://github.com/lvyiwei1/StylePTB)
  * [gpt-2-output-datase](https://github.com/openai/gpt-2-output-dataset)
  * [Parsing to FunQL](https://github.com/JasperGuo/Unimer)
  * [UKP](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345)
  * [NewsQuizQA](https://github.com/google-research-datasets/NewsQuizQA)
  * [Dialoglue](https://github.com/alexa/dialoglue)


## [UnifiedSKG](https://github.com/hkunlp/unifiedskg)

* Paper: [UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models](https://arxiv.org/pdf/2201.05966.pdf)
* Download: https://drive.google.com/drive/folders/1GXigUv3MU-Sh4XiY6Wz3xVeNT_s0SuON

## [MetaICL](https://github.com/facebookresearch/metaicl)

* Paper: [MetaICL: Learning to Learn In Context](https://arxiv.org/pdf/2110.15943.pdf)


## [openai/InstructionGPT](https://github.com/openai/following-instructions-human-feedback)

* Size: 112,801 instructions
* Language: EN
* Generation Method: Human Annotated
* Paper: [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)

## [facebookresearch/metasqe/OPT-IML](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT-IML)

* Size: 1,667 tasks, 3,128 instructions
* Language: EN
* Summary: OPT-IML dataset expands the Super-Natural-Instructions benchmark with the task collections from multiple existing work on instruction-tuning, cross-task transfer studies, and area-specific task consolidation.
* Generation Method:
  * Benchmarks included in OPT-IML are Super-Natural-Instructions, PromptSource, CrossFit, FLAN, ExMix, T5, UnifiedSKG, and Reasoning. Authors only kept partial tasks from CrossFit, ExMix and T5 due to the significant overlap.
  * To organize the Instruction schema, authors broadly classify the instructions in these benchmarks into two categories, dataset-level and instance-level.
* Paper: [OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://arxiv.org/pdf/2212.12017.pdf)
* License:

## [laion/OIG](https://laion.ai/blog/oig-dataset/)

* Size: 30 tasks, 43M instructions
* Language: EN
* Summary: OIG contains instructions that are created using data augmentation from a diverse collection of data sources, and formatted in a dialogue style (… … pairs).
* Generation Method:
  * OIG is created by various LAION community members, consisting of 30 datasets and 43M instructions, with the goal of reaching 1 trillion tokens.
  * OIG dataset can be divided roughly into 75% academic datasets, such as P3, Natural instructions and FLAN, and 25% datasets composed of various tasks, such as high school math, python coding and peoty generation.
* HuggingFace: https://huggingface.co/datasets/laion/OIG
* Demo: https://github.com/LAION-AI/Open-Assistant
* License:

## [baize/baize-chatbot](https://github.com/project-baize/baize-chatbot)

* Size: 3 tasks, 100K+ instructions
* Language: EN
* Summary: Baize dataset is a high-quality multi-turn chat corpus by leveraging ChatGPT to engage in a conversation with itself, named self-chatting.
* Generation Method:
  * First apply a template to define the format and requirements of a conversation.
  * Then use questions from Quora and Stack Overflow as seeds that set the topic for the chat.
* Paper: [Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data](https://arxiv.org/pdf/2304.01196.pdf)
* HuggingFace: (if applicable)
* Demo: https://huggingface.co/spaces/project-baize/Baize-7B
* License:

## [lightaime/camel](https://github.com/lightaime/camel)

* Size: 115K instructions
* Language: EN
* Summary: Camel dataset introduces a novel communicative agent framework named role-playing.
* Generation Method:
  * The prompt engineering in Camel consists of three prompts, the task specifier prompt, the assistant system prompt, and  the user system prompt. The scenarios in Camel include AI Society and Code.
  * Authors also create Data Generation Prompts to generate meta data by LLMs. 50 assistant roles and 50 user roles are generated for AI Society. 20 programming languages and 50 domains are generated for Code.
* Paper: [CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society](https://arxiv.org/pdf/2303.17760.pdf)
* HuggingFace: https://huggingface.co/camel-ai
* Demo: https://www.camel-ai.org/
* License:

## [thunlp/UltraChat](https://github.com/thunlp/UltraChat)

* Size: 657K instructions
* Language: EN
* Summary: UltraChat is a multi-round dialogue dataset powered by Turbo APIs, composed of three sectors, namely Questions about the World, Writing and Creation, and Assistance on Existent Materials.
* Generation Method:
  * Two separate ChatGPT Turbo APIs are adopted in generation, where one plays the role of the user to generate queries and the other generates the response. 
  * We instruct the user model with carefully designed prompts to mimic human user behavior and call the two APIs iteratively.
* HuggingFace: https://huggingface.co/datasets/stingning/ultrachat
* License:

## [databrickslabs/doll](https://github.com/databrickslabs/dolly)

* Size: 7 tasks, 15,000 instructions
* Language: EN
* Summary: Dolly is a human-generated corpus, whose categories are Creative Writing, Closed QA, Open QA, Summarization, Information Extraction, Classification and Brainstorming.
* Generation Method:
  * Databricks employees were invited to create prompt / response pairs in each of eight different instruction categories.
  * For instruction categories that require an annotator to consult a reference text, contributors selected passages from Wikipedia for particular subsets of instruction categories. 
* HuggingFace: https://huggingface.co/datasets/databricks/databricks-dolly-15k
* License:

## [stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP)

* Size: 18 tasks, 385K instructions
* Language: EN
* Summary: SHP is a dataset of 385K collective human preferences over responses to questions/instructions in 18 different subject areas, from cooking to legal advice. It is used to train RLHF reward models and NLG evaluation models.
* Generation Method:
  * The data is sourced from Reddit, which is a public forum organized into topic-specific fora called subreddits.
  * Each example is a Reddit post with a question/instruction and a pair of top-level comments for that post.
* Paper: [Understanding Dataset Difficulty with V
-Usable Information](https://proceedings.mlr.press/v162/ethayarajh22a/ethayarajh22a.pdf)
* HuggingFace: https://huggingface.co/datasets/stanfordnlp/SHP
* License:

## [Anthropic/hh-rlhf](https://github.com/anthropics/hh-rlhf)

* Size: 169,550 instructions
* Language: EN
* Summary: HH-RLHF is a dataset of human preferences over models' responses to questions/instructions.
* Generation Method:
  * Hire crowdworkers to interact with models through two interfaces, helpfulness interface and harmlessness (red-teaming) interface respectively.
  * For the helpfulness dataset, ask crowdworkers to have open-ended conversations with our models, asking for help, advice, or for the model to accomplish a task, and to choose the model response that was more helpful.
  * For the harmlessness (red-teaming) dataset, ask crowdworkers to attempt to elicit harmful responses from our models, and to choose the more harmful response offered by the models.
* Paper:
  * [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2204.05862.pdf)
  * [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/pdf/2209.07858.pdf)
* HuggingFace: https://huggingface.co/datasets/Anthropic/hh-rlhf
* License:

## [HuggingFaceH4/stack-exchange-preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)

* Size: 10M instructions
* Language: EN
* Summary: Stack-Exchange-Preferences dataset contains questions and answers from the Stack Overflow Data Dump for the purpose of preference model training.
* Generation Method:
* Paper: [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/pdf/2112.00861.pdf)
* HuggingFace: https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences
* License:

## [Hellp-SimpleAI/HC3](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)

* Size: 12 tasks, 37,175 instructions
* Language: EN, CH
* Summary: HC3 is a comparison corpus that consists of both human and ChatGPT answers to the same questions.
* Generation Method:
  * Human Answers Collection: The first part is publicly available question-answering datasets, whose answers are given by experts or high-voted. The second part is built by constructing question-answer pairs from wiki sources.
  * ChatGPT Answers Collection: use ChatGPT to generate answers to the questions in Human Answers Collection
* Paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/pdf/2301.07597.pdf)
* HuggingFace: https://huggingface.co/datasets/Hello-SimpleAI/HC3
* License: CC-BY-SA



# The Chinese Instruction Datasets

## [FlagOpen/FlagInstruct](https://github.com/flagopen/flaginstruct)
* Size: 2K tasks, 191,191 instructions in total
* Language: CH
* Summary: Chinese Open Instruction Generalist (COIG) is a Chinese instruction dataset consisting of 4 sub-tasks.
* Generateion Method:
  * Task 1: Translated Instructions (67,798)
    * Translate the following datasets into Chinese: 1,616 task descriptions in Super-Natural-Instruct v2 along with a single instance for each of them; 175 seed tasks in Self-instruct; 66,007 instructions from Unnatural Instructions.
  * Task 2: Exam Instructions (63,532)
    * Exams include The Chinese National College Entrance Examination (高考), Middle School Entrance Examinations (中考), and Civil Servant Examination (公务员考试).
    * Turn them into Chain-of-Thought (CoT) corpus by extracting six informative elements from original exam questions, including instruction, question context, question, answer, answer analysis, and coarse-grained subject.
  * Task 3: Human Value Alignment Instructions (34,471)
    * Select a set of samples that present shared human values in the Chinese-speaking world, and get 50 seed instructions and 3k resulting instructions.
    * Some additional sets of samples that present regional-culture or country-specific human values are also added.
  * Task 4: Counterfactural Correction Multi-round Chat (13,653)
    * The aim is to alleviate and resolve the pain points of hallucination and factual inconsistency in current LLMs.
    * Based on [CN-DBpedia knowledge graph dataset](https://link.springer.com/chapter/10.1007/978-3-319-60045-1_44), CCMC has ~13,000 dialogues with an average of 5 rounds per dialogue, resulting in ~65,000 rounds of chat.
  * Leetcode Instructions (11,737)
    * 2,589 programming questions from [Leetcode](https://github.com/doocs/leetcode).
* Paper: [Chinese Open Instruction Generalist: A Preliminary Release](https://arxiv.org/pdf/2304.07987.pdf)
* HuggingFace: https://huggingface.co/datasets/BAAI/COIG
* License: MIT License

## [owner/project-name](https://github.com/link/to/project)

* Size:
* Language:
* Summary:
* Generation Method:
* Paper:
* HuggingFace: (if applicable)
* Demo: (if applicable)
* License:


## Chinese Alpaca

### [carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset) 
* Size: 20,456 instructions
* Language: CH
* Generateion Method: Translate Alpaca into Chinese by machine and then clean.

### [hikariming/alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset) 
* Size: 19,442 instructions
* Language: CH
* Generateion Method: Translate Alpaca into Chinese by ChatGPT, and check them by humans

### [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 
* Size: 51,458 instructions
* Language: CH
* Generateion Method: Translate Alpaca into Chinese by ChatGPT, and discard some of them.

### [LC1332/Chinese-alpaca-lora](https://github.com/LC1332/Chinese-alpaca-lora) 
* Size: 51,672 instructions
* Language: CH
* Generateion Method: Translate Stanford Alpaca dataset into Chinese by ChatGPT.

### [A-baoYang/alpaca-7b-chinese](https://github.com/A-baoYang/alpaca-7b-chinese) 
* Size: 20,465 instructions
* Language: TC
* Generateion Method: Translate Stanford Alpaca dataset into traditional Chinese using OpenCC.

### [A-baoYang/alpaca-7b-chinese](https://github.com/A-baoYang/alpaca-7b-chinese) 
* Size: 124,469 instructions
* Language: EN, TC
* Generateion Method: Combine the English instruction/input and traditional Chinese output by ChatGPT.

### [ntunlplab/traditional-chinese-alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca) 
* Size: 52,002 instructions
* Language: EN, TC
* Generateion Method: A Traditional-Chinese version of the Alpaca dataset, whose instruction part is left as English.

### [ntunlplab/traditional-chinese-alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca) 
* Size: 52,002 instructions
* Language: EN, TC
* Generateion Method: An Traditional-Chinese version of the Alpaca dataset, where there are English and traditional Chinese versions of one single instruction.

# The Miltilingual Instruction Datasets

## [bigscience/xP3](https://github.com/bigscience-workshop/xmtf) 
* Size: 83 tasks
* Language: Multilingual (46 languages)
* Summary: 
  * xP3 is a mixture of 13 training tasks in 46 languages with English prompts. 
  * Moreover, there is a xP3 Dataset Family, including the following two datasets:
    * xP3mt is a mixture of 13 training tasks in 46 languages with prompts in 20 languages; 
    * xP3all consists of xP3 itself and evaluation datasets adding an additional 3 tasks.
* Generateion Method: Build on the P3 task taxonomy and add 28 new multilingual datasets.
* Paper: [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/pdf/2211.01786.pdf)
* HuggingFace: https://huggingface.co/datasets/bigscience/xP3
* License:

## [JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) 
* Size: 380,835 instructions in total
* Language: CH, DE, EN, JA, TC
* Summary: Guanaco dataset builds upon the 175 tasks from Alpaca, containing 3 versions with different sizes and methods.
* Generateion Method:
  * Original Version (48967): Rewrite 175 Alpaca seed tasks in different languages, and add new tasks specifically designed for English grammar analysis, natural language understanding, cross-lingual self-awareness, and explicit content recognition.
  * Mixed Version (279644): The original 175 tasks were translated into 4 versions and regenerated independently, excluding Deutsch.
  * MIni Version (52224): 52K instrucrion dataset, which is included in the Mixed Version.
* HuggingFace: https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/tree/main
* License:

## [JosephusCheung/GuanacoDataset QA](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) 
* Size: 205,999 instructions in total
* Language: CH, DE, EN, JA
* Summary: The Paper/General-QA dataset is a collection of questions and answers constructed for AI-generated papers or general texts in 4 languages. The purpose of this dataset is to generate paragraph-level answers to questions posed about lengthy documents such as PDFs. 
* Generateion Method:
  * The question dataset contains 106,707 questions, and the answer dataset contains 99,292 answers.
  * Similar questions are combined to form a tree-like structure, and graph theory algorithms are used to process user questions, content summaries, and contextual logic.
* HuggingFace: https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/tree/main/additional
* License:


# Former Tables (to be removed)

| Dataset | Size | Language | Domain | Generation method |
|:---------| :---------:|:---------:|:---------:|:---------|
| [hikarming/alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset/tree/main/%E5%85%B6%E4%BB%96%E4%B8%AD%E6%96%87%E9%97%AE%E9%A2%98%E8%A1%A5%E5%85%85) | 226 | CH | topic-specific | Generate Chinese instructions under various topics by humans, such as bussiness management, education, Romance of the Three Kingdoms, etc. |
| [sahil280114/codealpaca](https://github.com/sahil280114/codealpaca) | 20023 | EN | Code | Self-instuct with prompts to focus on code generation/edting/optimization tasks, using text-davinci-003. |
| [QingyiSi/Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) | | | | |
