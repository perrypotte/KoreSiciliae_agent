import asyncio
import json
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def main():
   """  # 1. Define the extraction schema
    schema = {
        "name": "Product List",
        "baseSelector": "div.product-card",  # The repeating container for each item
        "fields": [
            {"name": "title", "selector": "h2", "type": "text"},
            {"name": "price", "selector": ".price-tag", "type": "text"},
            {"nameP": "link", "selector": "a", "type": "attribute", "attribute": "href"}
        ]
    }

    # 2. Create the non-LLM strategy
    strategy = JsonCssExtractionStrategy(schema)

    # 3. Run the crawler
    config = CrawlerRunConfig(extraction_strategy=strategy)
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://example.com/products", config=config)
        
        if result.success:
            # result.extracted_content contains your structured JSON string
            data = json.loads(result.extracted_content)
            print(json.dumps(data, indent=2)) """
   async with AsyncWebCrawler() as crawler:
    # result = await crawler.arun(
    #    url="https://www.koresiciliae.it/ci-presentiamo",
    #    actions=[{"action": "wait_for", "selector": ".activity-category-item.link-reset"},
    #     {"action": "click", "selector": ".activity-category-item.link-reset"},
    #     {"action": "wait_for", "selector": ".activity-theme-item-icon"},
    #     {"action": "screenshot", "path": "debug.png"}
    # ]
    # )

    #TODO estrarre gli indici di tutti i viewSource della pagina sembra essere la migliore opzione per automatizzare tra virgolette.
    #TODO fare una queue per le pagine già visitate e quelle da visitare
    #TODO pulire il markdown da immagini e link (????)
    #TODO Le risorse singole posso ricavare i metadati del tipo direttamente dai titoli che hanno la stessa classe
    #TODO
    result = await crawler.arun(
    url="https://www.koresiciliae.it/ci-presentiamo",
    actions=[
        {
            "action": "evaluate",
            "script": """
            return Array.from(document.querySelectorAll('[onclick]'))
                .map(el => el.getAttribute('onclick'))
            """
        }
            ]
        )

    actions_list = result.js_result
    print(actions_list)
    for i in range(1):
        result = await crawler.arun(
            url="https://www.koresiciliae.it/ci-presentiamo",
            actions=[
                {"action": "wait_for", "selector": "[onclick]"},
                {
                    "action": "evaluate",
                    "script": f"document.querySelectorAll('[onclick]')[{i}].click()"
                },
                {"action": "wait", "seconds": 2}
            ],
            #extraction_strategy="markdown"
        )
    
    # Standard Markdown version of the page
    #print(result.markdown)
    
   # QUA SERVONO I FILTRI DA DEFINIRE, VEDI DOCS
    # "Fit" Markdown: Highly pruned content (useful for RAG or reading)
    # print(result.markdown.fit_markdown)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=70,
        #separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(result.markdown)
    #print(f"links found: {result.links}")
    print(f"Total chunks created: {len(chunks)}")
    print("Sample chunk:")
    for i, chunk in enumerate(chunks[:12]):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print("\n")

    

if __name__ == "__main__":
    asyncio.run(main())