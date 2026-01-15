import html
import json
import re
import time
from textwrap import dedent
from typing import Dict, Iterator, Optional
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from agno.utils.log import logger
from agno.workflow import RunResponse, Workflow

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
        self.query_planner = blog_agents.query_planner_agent
        self.writer = blog_agents.writer_agent

    def _sanitize_user_text(self, text: str, max_length: int = 1200) -> str:
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:max_length]

    def _should_use_query_planner(self, raw_topic: str) -> bool:
        return len(raw_topic) > 120 or "\n" in raw_topic or raw_topic.count(".") > 1

    def _build_search_query(self, raw_topic: str, cleaned_topic: str, style_guidelines: str) -> str:
        if not self.query_planner or not self._should_use_query_planner(raw_topic):
            return cleaned_topic
        try:
            planner_input = json.dumps(
                {
                    "topic": cleaned_topic,
                    "style_guidelines": style_guidelines,
                },
                indent=2,
            )
            response: RunResponse = self.query_planner.run(planner_input)
            if response and response.content:
                query = self._sanitize_user_text(str(response.content), max_length=200)
                query = query.strip().strip('"')
                if query:
                    return query
        except Exception as exc:
            logger.warning(f"Failed to generate search query: {exc}")
        return cleaned_topic

    def _search_google_news_rss(self, topic: str, max_results: int = 10) -> SearchResults:
        query = quote_plus(topic)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AI-Blog-Generator/1.0; +https://example.com)"
            },
        )
        with urlopen(request, timeout=10) as response:
            feed = response.read().decode("utf-8")

        articles: list[NewsArticle] = []
        seen_urls: set[str] = set()
        try:
            root = ElementTree.fromstring(feed)
            items = root.findall(".//item")
            for item in items:
                title = item.findtext("title")
                link = item.findtext("link")
                description = item.findtext("description")
                if not title or not link:
                    continue
                title = html.unescape(title).strip()
                url = link.strip()
                summary_raw = html.unescape(description).strip() if description else None
                summary = re.sub(r"<[^>]+>", "", summary_raw).strip() if summary_raw else None
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append(NewsArticle(title=title, url=url, summary=summary))
                if len(articles) >= max_results:
                    break
        except ElementTree.ParseError:
            items = re.findall(r"<item>(.*?)</item>", feed, flags=re.DOTALL)
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
        return SearchResults(articles=articles[:max_results])

    def get_search_results(self, topic: str, num_attempts: int = 3) -> Optional[SearchResults]:
        for attempt in range(num_attempts):
            try:
                search_results = self._search_google_news_rss(topic)
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

            article_scraper_response: RunResponse = self.article_scraper.run(article)
            if (
                article_scraper_response is not None
                and article_scraper_response.content is not None
                and isinstance(article_scraper_response.content, ScrapedArticle)
            ):
                scraped_articles[article_scraper_response.content.url] = article_scraper_response.content
                logger.info(f"Scraped article: {article_scraper_response.content.url}")
                if not article_scraper_response.content.content:
                    logger.warning(f"No readable content found for article: {article.url}")
            else:
                logger.warning(f"Skipping article due to scrape failure: {article.url}")
        return scraped_articles

    def run(
        self,
        topic: str,
        style_guidelines: Optional[str] = None,
        include_sources: bool = True,
    ) -> Iterator[RunResponse]:
        """Run the blog post generation workflow."""
        raw_topic = topic or ""
        cleaned_topic = self._sanitize_user_text(raw_topic, max_length=600)
        if not cleaned_topic:
            yield RunResponse(content="Please provide a topic or some ideas to get started.")
            return
        cleaned_guidelines = self._sanitize_user_text(style_guidelines or "", max_length=1500)
        logger.info(f"Generating a blog post on: {cleaned_topic}")

        # Search the web for articles on the topic
        search_query = self._build_search_query(raw_topic, cleaned_topic, cleaned_guidelines)
        search_results: Optional[SearchResults] = self.get_search_results(search_query)
        # If no search_results are found for the topic, end the workflow
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                content=f"Sorry, could not find any articles on the topic: {cleaned_topic}",
            )
            return

        # Scrape the search results
        scraped_articles: Dict[str, ScrapedArticle] = self.scrape_articles(cleaned_topic, search_results)
        if not scraped_articles:
            logger.warning("No articles scraped successfully. Falling back to RSS summaries only.")
            scraped_articles = {
                article.url: ScrapedArticle(
                    title=article.title,
                    url=article.url,
                    summary=article.summary,
                    content=None,
                )
                for article in search_results.articles
            }

        # Prepare the input for the writer
        writer_input = {
            "topic": cleaned_topic,
            "style_guidelines": cleaned_guidelines,
            "include_sources": include_sources,
            "articles": [v.model_dump() for v in scraped_articles.values()],
        }

        # Run the writer and yield the response
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
