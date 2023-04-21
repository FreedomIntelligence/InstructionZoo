# InstructionZoo

A collection of open-source Instruction-tuning dataset to train chat-based LLMs (ChatGPT,LLaMA,Alpaca).

This is an on-going project. The format and explaination of the following contents will be updated soon. (By Zhihan)

# Table of Contents

# The template

```
## [({owner}/{project-name)}]{https://github.com/link/to/project}

- Size:
- Language: 
- Summary:
- Generateion Method:
- Paper:
- HuggingFace: (if applicable)
- Demo: (if applicable)
- License:
```

# The English and Miltilingual Instruction Datasets

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


# Former Tables

## Instruction Dataset Collection

| Dataset | Size | Language | Domain | Generation method |
|:---------| :---------:|:---------:|:---------:|:---------|
| [hikarming/alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset/tree/main/%E5%85%B6%E4%BB%96%E4%B8%AD%E6%96%87%E9%97%AE%E9%A2%98%E8%A1%A5%E5%85%85) | 226 | CH | topic-specific | Generate Chinese instructions under various topics by humans, such as bussiness management, education, Romance of the Three Kingdoms, etc. |
| [sahil280114/codealpaca](https://github.com/sahil280114/codealpaca) | 20023 | EN | Code | Self-instuct with prompts to focus on code generation/edting/optimization tasks, using text-davinci-003. |
| [XueFuzhao/InstructionWild](https://github.com/XueFuzhao/InstructionWild) | 52191 (479 seeds) | CH, EN | | Collect 429 instructions from ChatGPT usage screenshots and release both English and Chinese versions, using text-davinci-003. |
| [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | 500000 (175 seeds) | CH | | Self-instruct with 175 Chinese seed tasks translated from the seed tasks in Stanford Alpaca dataset, using text-davinci-003. |
| [BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN) | 1000000 (175 seeds) | CH | | Self-instruct with 175 Chinese seed tasks translated from the seed tasks in Stanford Alpaca dataset. |
| [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | 250000 | CH | Math | Chinese math questions and answers generated by ChatGPT. |
| [BelleGroup/multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) | 800000 | CH | Multiturn Chat | Instruction contains historical dialog context, distinguishable by Human: and Assistant:, output contains the current reply by assistant. |
| [GuanacoDataset/.../guanaco_chat_all-utf8.json](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/blob/main/guanaco_chat_all-utf8.json) | 48967 | CH, DE, EN, JA, TC | Multiturn Chat, Multi-lingual | The dataset for the Guanaco model  builds upon the 175 tasks from the Alpaca model by providing rewrites of seed tasks in different languages and adding new tasks specifically designed for English grammar analysis, natural language understanding, cross-lingual self-awareness, and explicit content recognition. |
| [GuanacoDataset/.../guanaco_non_chat-utf8.json](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/blob/main/guanaco_non_chat-utf8.json) | 279644 | CH, DE, EN, JA, TC | Multi-lingual | The original 175 tasks were translated into 4 versions and regenerated independently. |
| [GuanacoDataset/.../guanaco_non_chat_mini_52K-utf8.json](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/blob/main/guanaco_non_chat_mini_52K-utf8.json) | 52224 | CH, DE, EN, JA, TC | Multi-lingual | A mini version of 52K multi-lang dataset. |
| [GuanacoDataset/.../general_ans-utf8.json](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/blob/main/additional/general_ans-utf8.json) | 75899 | CH, DE, EN, JA, TC | paragraph-level QA, Multi-lingual | |
| [GuanacoDataset/.../general_questions-utf8.json](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/blob/main/additional/general_questions-utf8.json) | 82867 | CH, DE, EN, JA, TC | paragraph-level QA, Multi-lingual | Similar questions are combined to form a tree-like structure, and graph theory algorithms are used to process user questions, content summaries, and contextual logic. |
| [GuanacoDataset/.../paper_answers-utf8.json](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/blob/main/additional/paper_answers-utf8.json) | 23393 | CH, DE, EN, JA, TC | paragraph-level QA, paper QA, Multi-lingual | |
| [GuanacoDataset/.../paper_questions-utf8.json](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset/blob/main/additional/paper_questions-utf8.json) | 23840 | CH, DE, EN, JA, TC | paragraph-level QA, paper QA, Multi-lingual | |
| [PhoebusSi/alpaca-CoT](https://github.com/PhoebusSi/alpaca-CoT) | EN | Chain-of-Thought | | |
| [QingyiSi/Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) | | | | |
