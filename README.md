# Create your AI or LLM Strategy

Large-scale language models like GPT-4 from OpenAI serve as foundational technologies that can be applied to virtually any business issue. However, the robust power and flexibility of this technology come with a significant challenge: it is extremely difficult to pinpoint the optimal opportunities for leveraging this technology within a company.

This project is designed to assist analytics leaders, product managers, and development teams in surmounting these obstacles by demonstrating the technology's application across a variety of common business problems. The project unfolds through a series of episodes, each accompanied by the following resources:

1. Walkthrough videos available on [Prolego's YouTube Channel](https://www.youtube.com/@prolegoinc).
2. Tagged releases on the main branch of this repository.
3. Conversations held within [Prolego's Discord community](https://discord.gg/eaPAhJcTpJ).

## Capabilities approach
We advise our clients to take a capabilties-based approach when building their AI. That is, create foundational solutions that allow you to solve many different business use cases. Unfortunately too many teams begin solving specifing business problems withough building a generalizable foundation.

Most companies are developing the following capabilties as part of their AI strategy.

| Capability                     | Explanation                                                                                                                  | Examples                                                               |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| text classification            | Assigning categories to documents or document sections.                                                                      | [Episode 2](#episode-2---automated-document-processing)                |
| information extraction         | Pulling out names, places, or specific sections from documents.                                                              | [Episode 2](#episode-2---automated-document-processing)                |
| semantic search                | Finding information based on its meaning instead of keywords.                                                                | [Episode 1](#episode-1---embeddings-semantic-search-and-document-qa)   |
| information summarization      | Condense extensive documents into concise and essential highlights.                                                          |                                                                        |
| information comparison         | Identifying similar documents or sections of documents.                                                                      |                                                                        |
| document generation            | Creating precisely written content consistent with style and needs. Often includes a review step.                              | [Episode 2](#episode-2---automated-document-processing)                |
| unified natural language query | Empowering anyone to get answers to questions about data and documents without SQL or tools.                                   | [Episode 3](#episode-3---intro-to-unified-natural-language-query), [Episode 4](#episode-4---unified-nlq-with-a-table-and-document), [Episode 5](#episode-5---unified-nlq-with-multiple-tables) |
| routine task automation        | Automating analysis of information from various sources, reasoning across them, and making decisions.                         |       [Episode 4](#episode-4---unified-nlq-with-a-table-and-document)                                                                 |

## Quickstart installation
Remember, this is demo code! Don't attempt to use it for production without first redesigning it.

First install the neo-sophia code on your local machine before proceeding to the examples from the Episodes below.

### Install the base neo-sophia code

    git clone https://github.com/prolego-team/neo-sophia.git
    conda env create -f neo-sophia/env.yml
    conda activate neosophia
    pip install -e neo-sophia
    cd neo-sophia
    cp config_example.json config.json
    cp openai_api_key_example.txt openai_api_key.txt

### Setup your local configuration.

1. Change the path locations in `config.json` or use the defaults.
2. Add your OpenAI API key to `openai_api_key.txt`.

### Run the tests.

    ./test.sh

If the tests pass you are ready to run the code in one of the Episodes.

Questions? Just ask in our [Discord Community](https://discord.gg/eaPAhJcTpJ).

# Episodes

## Episode 19 - Demonstrating RAG benefits for customers and constituents using GPTs

Video: [Build a RAG demo in 1 hour with GPTs](https://youtu.be/3mIUFSylsgY)

## Episode 18 - Discover AI Opportunities with Generated Data

Video: [Discover AI Opportuniteis with generated data](https://youtu.be/Ul-22gj5GzU)

## Episode 17 - Intro to Retrieval Augmented Generation

Video: [Intro to RAG](https://youtu.be/LmiFeXH-kq8)

1. Checkout Episode 17, [Release v0.17.0](https://github.com/prolego-team/neo-sophia/releases/tag/v0.17.0)
````
    git checkout tags/v0.17.0
````
2. Start the demo by running

````
    python -m examples.fia.reg_search

````

## Episode 16 - Automated document processing case study
What are companies doing with Generative AI? Here's how one CEO did it.

Video: [How Vericant released its Generative AI solution in 30 days](https://youtu.be/Sz_F8p2fBBk)

## Episode 15 - Should You Use Open Source LLMs or GPT-4?

Video: [Should You Use Open Source LLMs or GPT-4?](https://www.youtube.com/watch?v=YvwPoz9hAq4&t=11s)

Running this code requires additional steps by an experienced developer. See [the pull request](https://github.com/prolego-team/neo-sophia/pull/116) 

## Episode 14 - The three most common entperise LLM projects

Video: [The Top 3 Enterprise AI Use Cases](https://youtu.be/b6k8-mqpmrc?si=OnMUDl35rMPmKLG8)

## Episode 13 - Software design pattern for LLM Scratchpads
In the second episode on Scratchpads we walk through the technical detaials and workflow.

Video: [Cook a Scrumptious AI Solution with LLM Scratchpads](https://www.youtube.com/watch?v=YLTronn1riw)

Same demo as Episode 12.

## Episode 12 - Data too big for LLMs? Try Scratchpads
LLMs cannot handle data bigger than their context windows. To overcome these limitations, use temporary memory called LLM Scratchpads.

1. Checkout Episode 12, [Release v0.12.0](https://github.com/prolego-team/neo-sophia/releases/tag/v0.12.0)
````
    git checkout tags/v0.12.0
````
2. Start the demo by running

````
    python -m examples.agent_example

````

## Episode 11 - Don't use LangChain (yet)
Generalized LLM frameworks like LangChain are not mature enough for most teams. The foundational models and best practices are changing too fast to generalize. 

Video: [LangChain and other frameworks are not ready](https://www.youtube.com/watch?v=_BHaYRujkDI)

## Episode 10 - Turn your dashboards into interactive data conversations
Dashboards and reports written with applications like Tableau are underutilized. Fortunately you can turn the into rich conversations with LLMs.

Video: [Hate Tableau? Replace it with LLMs](https://www.youtube.com/watch?v=tRTlbDc0i2E)

1. Checkout Episode 10, [Release v0.10.0](https://github.com/prolego-team/neo-sophia/releases/tag/v0.10.0)
````
    git checkout tags/v0.10.0
````
2. Start the demo by running

````
    python -m examples.sleuth
````

## Episode 9 - Build your AI Strategy in One Hour
Most companies spend months talking about AI before converging on the same plan as everyone else. Save yourself the time and start with the right plan.

Video:[Write Your Company's Generative AI Strategy in ONE hour](https://www.youtube.com/watch?v=0vvOQWoSc_o&list=PL-pTHQz4RcBb6kiMuMg5W2c2Yu_-ij27s)

## Episode 8 - The critical skills needed for LLM development
Programming against LLM APIs requires a combination of an experimental mindset and systems engineering experience.

Video: [The skills needed for LLM Developent](https://www.youtube.com/watch?v=Wp6LP8yk4I8)
Job description: [AI Systems Engineer](https://www.prolego.com/email/aiengineerjobdescription)

## Episode 7 - How to pick the Best LLM for Your Project
GPT-3? GPT-4? Claude 2? Open source? You have so many options and don't know where to start. Fortunately, there is an easy answer for the majority of teams. Start with the most intelligent model, currently GPT-4.

Video: [How to pick the Best LLM for Your Project](https://www.youtube.com/watch?v=VTWn-2SVWo8))

## Episode 6 - Improving LLM Reliablity with Evaluation frameworks
LLM hallucinations and inconsistencies are real challenges, but you can begin overcoming them with a good evaluation framework.

Video: [Conquer LLM Hallucinations with an Evaluation Framework](https://www.youtube.com/watch?v=mW3_vn1zGfE)

1. Checkout Episode 6, [Release v0.6.0](https://github.com/prolego-team/neo-sophia/releases/tag/v0.6.0)
````
    git checkout tags/v0.6.0
````
2. Start the demo by running

````
    python -m examples.bank_agent_eval
````

## Episode 5 - Queries Across Multiple Data Sources
Every practical problem you encounter will require accessing large datasets such as multiple databases. In doing so you will encounter the limits of the LLM's context window. In this example we explain this limitation and a simple approach for overcoming it.

Video: [How to Overcome LLM Context Window Limitations](https://www.youtube.com/watch?v=_deDqraxqog)

1. Checkout Episode 5, [Release v0.5.0](https://github.com/prolego-team/neo-sophia/releases/tag/v0.5.0)
````
    git checkout tags/v0.5.0
````
2. Start the demo by running

````
    python  examples/bank_agent_two.py
````


## Episode 4 - Intro to LLM Agents, querying multiple tables
In our second episode on Unified NLQ we introduce LLM Agents. Agents are necessary for the complex reasoning required for to run natural language queries across multiple tables.

Video: [Supercharge Your Data Anlaytics with LLM Agents](https://www.youtube.com/watch?v=XV4IBaZqbps)

1. Checkout Episode 4, [Release v0.4.0](https://github.com/prolego-team/neo-sophia/releases/tag/v0.4.0)
````
    git checkout tags/v0.4.0
````
2. Start the demo by running

````
    python -m examples.bank_agent
````

## Episode 3 - Intro to Unified Natural Language Query
What will be the AI "killer app" in the enterprise? Our bet is Unified Natural Language Query (NQL). It gives executives and business leaders the ability to get insights from data by asking "natural" questions, similar to how you currently use ChatGPT. In this Episode we describe the business problem and show the extensible power of a simple example of SQL generation supplemented with the reasoning power of an LLM like GPT-4.

Video
- [Unified Natural Language Query is the enerprise AI "killer app"](https://www.youtube.com/watch?v=zuLxXDdEVEE)

1. Checkout Episode 3, [Release v0.3.2](https://github.com/prolego-team/neo-sophia/releases/tag/v0.3.2)

````
    git checkout tags/v0.3.2
````
2. Start the demo by running

````
    python -m examples.sqlite_chat
````

## Episode 2 - Automated document processing
Every company has businesses processes that require ingesting and processing a stream of text documents. Most of this processing requires tedious human effort to find, edit, review, summarize, score, etc. chunks of text from larger documents. In this Episode we demonstrate a generalized approach for solving many of these problems using LLMs. The example takes a set of SEC 10-Q company filings and replaces the "Basis of Presentation" section with different text based on an editable templates.

Videos
- [Your AI strategy “quick win” - automated document processing](https://www.youtube.com/watch?v=Ba1yc5Sh9UE)
- [Audomated document processing - technical walkthrough](https://www.youtube.com/watch?v=_9gBQ4nDZUw)

### Running the Episode 2 application

1. Checkout Episode 2, [Release v0.2.1](https://github.com/prolego-team/neo-sophia/releases/tag/v0.2.1)

````
    git checkout tags/v0.2.1
````

2. Start the demo by running
````
    python -m examples.generate_10q_basis
````

## Episode 1 - Embeddings, Semantic search and Document Q&A
Most companies are struggling to pick the best AI use cases from many different options. By building a core competency in document embeddings you can begin developing a set of capabilities applicable for many enterprise use cases. In Episoide 1 we provide a primer on embeddings for a business audience and demonstrate the use of embeddings in semantic search and document Q&A.

This episode uses data from the [MSRB Regulatory Rulebook](https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf)

Videos
- [Document embeddings are foundational capabilities for your AI strategy](https://www.youtube.com/watch?v=EQx-iTjQClM)
- [Document embeddings - technical walkthrough](https://www.youtube.com/watch?v=RfGOvkGj678)

### Running the Episode 1 application

1. Checkout Episode 1, [Release v0.1.1](https://github.com/prolego-team/neo-sophia/releases/tag/v0.1.1)
````
    git checkout tags/v0.1.1
````
2. Start the demo by running
````
    python -m examples.interface
````

# About Prolego & Neo Sophia

[Prolego](https://prolego.com) is an AI services company that started in 2017 and has helped some of the world’s biggest companies generate opportunities with AI. "Prolego" is the Greek word for "predict". We needed a name for this repo and decided to use the Greek words for "new" (neo) and "wisdom" (sophia). And we just thought that Neo Sophia sounded cool.

The team:

* [Kevin Dewalt](https://github.com/kevindewalt)
* [Justin Pounders](https://github.com/jmpounders)
* [Cameron Fabbri](https://github.com/cameronfabbri)
* [Ben Zimmer](https://github.com/bdzimmer)
