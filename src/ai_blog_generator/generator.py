import html
import json
import re
import time
from textwrap import dedent
from typing import Dict, Iterator, Optional
from urllib.parse import quote_plus
from urllib.request import urlopen

from duckduckgo_search import DDGS

from agno.utils.log import logger
from agno.workflow import RunEvent, RunResponse, Workflow

from ai_blog_generator.response_model import NewsArticle, ScrapedArticle, SearchResults


class BlogPostGenerator(Workflow):
    """Advanced workflow for generating professional blog posts with proper research and citations."""

    description: str = dedent("""\
    An intelligent blog post generator that creates engaging, well-researched content.
    This workflow orchestrates multiple AI agents to research, analyze, and craft
    compelling blog posts that combine journalistic rigor with engaging storytelling.
    The system excels at creating content that is both informative and optimized for
    digital consumption.
    """)

    def __init__(self, blog_agents, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.article_scraper = blog_agents.article_scraper_agent
        self.writer = blog_agents.writer_agent

    def _search_google_news_rss(self, topic: str, max_results: int = 10) -> SearchResults:
        query = quote_plus(topic)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        with urlopen(url, timeout=10) as response:
            feed = response.read().decode("utf-8")

        items = re.findall(r"<item>(.*?)</item>", feed, flags=re.DOTALL)
        articles: list[NewsArticle] = []
        seen_urls: set[str] = set()

        for item in items:
            title_match = re.search(r"<title><!\\[CDATA\\[(.*?)\\]\\]></title>", item)
            link_match = re.search(r"<link>(.*?)</link>", item)
            desc_match = re.search(r"<description><!\\[CDATA\\[(.*?)\\]\\]></description>", item)
            title = html.unescape(title_match.group(1)).strip() if title_match else None
            url = link_match.group(1).strip() if link_match else None
            summary_raw = html.unescape(desc_match.group(1)).strip() if desc_match else None
            summary = re.sub(r"<[^>]+>", "", summary_raw).strip() if summary_raw else None
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(NewsArticle(title=title, url=url, summary=summary))
            if len(articles) >= max_results:
                break

    def _search_with_duckduckgo(self, topic: str, max_results: int = 12) -> SearchResults:
        query = f"{topic} insights analysis"
        articles: list[NewsArticle] = []
        seen_urls: set[str] = set()
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                url = result.get("href") or result.get("url")
                title = result.get("title") or result.get("heading")
                summary = result.get("body") or result.get("snippet") or result.get("description")
                if not url or not title or url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append(NewsArticle(title=title, url=url, summary=summary))
        return SearchResults(articles=articles[:7])

    def get_search_results(self, topic: str, num_attempts: int = 3) -> Optional[SearchResults]:
        for attempt in range(num_attempts):
            try:
                search_results = self._search_google_news_rss(topic)
                search_results = self._search_with_duckduckgo(topic)
                article_count = len(search_results.articles)
                if article_count > 0:
                    logger.info(f"Found {article_count} articles on attempt {attempt + 1}")
                    return search_results
                logger.warning(f"Attempt {attempt + 1}/{num_attempts} failed: No articles found")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
                time.sleep(min(2 * (attempt + 1), 5))

        logger.error(f"Failed to get search results after {num_attempts} attempts")
        return None

    def scrape_articles(self, topic: str, search_results: SearchResults) -> Dict[str, ScrapedArticle]:
        scraped_articles: Dict[str, ScrapedArticle] = {}
        for article in search_results.articles:
            if article.url in scraped_articles:
                logger.info(f"Found scraped article in cache: {article.url}")
                continue

            article_scraper_response: RunResponse = self.article_scraper.run(article.url)
            if (
                article_scraper_response is not None
                and article_scraper_response.content is not None
                and isinstance(article_scraper_response.content, ScrapedArticle)
            ):
                scraped_articles[article_scraper_response.content.url] = article_scraper_response.content
                logger.info(f"Scraped article: {article_scraper_response.content.url}")
        return scraped_articles

    def run(
        self,
        topic: str,
    ) -> Iterator[RunResponse]:
        """Run the blog post generation workflow."""
        logger.info(f"Generating a blog post on: {topic}")

        # Search the web for articles on the topic
        search_results: Optional[SearchResults] = self.get_search_results(topic)
        # If no search_results are found for the topic, end the workflow
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}",
            )
            return

        # Scrape the search results
        scraped_articles: Dict[str, ScrapedArticle] = self.scrape_articles(topic, search_results)

        # Prepare the input for the writer
        writer_input = {
            "topic": topic,
            "articles": [v.model_dump() for v in scraped_articles.values()],
        }

        # Run the writer and yield the response
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
