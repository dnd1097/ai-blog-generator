from textwrap import dedent

from newspaper import Article

from agno.agent import Agent
from agno.workflow import RunResponse

from ai_blog_generator.model import Model
from ai_blog_generator.response_model import NewsArticle, ScrapedArticle


class ArticleScraperAgent:
    """Scrapes article content without invoking LLM tool calls."""

    def run(self, article: NewsArticle) -> RunResponse:
        try:
            parsed_article = Article(article.url)
            parsed_article.download()
            parsed_article.parse()
            title = parsed_article.title.strip() if parsed_article.title else article.title
            content = parsed_article.text.strip() if parsed_article.text else None
            scraped_article = ScrapedArticle(
                title=title,
                url=article.url,
                summary=article.summary,
                content=content,
            )
            return RunResponse(content=scraped_article)
        except Exception as exc:
            return RunResponse(
                content=ScrapedArticle(
                    title=article.title,
                    url=article.url,
                    summary=article.summary,
                    content=None,
                ),
                error=str(exc),
            )


class BlogAgents:
    """Agents for blog post generation workflow"""

    def __init__(self, llm: Model):
        """Initialize the agents for blog post generation workflow"""
        self.llm = llm.get()
        self.article_scraper_agent = self._create_article_scraper_agent()
        self.writer_agent = self._create_writer_agent()

    # Content Scraper: Extracts and processes article content
    def _create_article_scraper_agent(self) -> Agent:
        """Create the article scraper agent for extracting content from articles"""
        return ArticleScraperAgent()

    # Content Writer Agent: Crafts engaging blog posts from research
    def _create_writer_agent(self) -> Agent:
        """Create the content writer agent for generating blog posts"""
        return Agent(
            model=self.llm,
            description=dedent("""\
         You are BlogMaster-X, an elite content creator combining journalistic excellence
         with digital marketing expertise. Your strengths include:

         - Crafting viral-worthy headlines
         - Writing engaging introductions
         - Structuring content for digital consumption
         - Incorporating research seamlessly
         - Optimizing for SEO while maintaining quality
         - Creating shareable conclusions\
         """),
            instructions=dedent("""\
         1. Content Strategy 📝
            - Craft attention-grabbing headlines
            - Write compelling introductions
            - Structure content for engagement
            - Include relevant subheadings
            - 800-1200 words per post
         2. Writing Excellence ✍️
            - Balance expertise with accessibility
            - Use clear, engaging language
            - Include relevant examples
            - Incorporate statistics naturally
         3. Source Integration 🔍
            - Cite sources properly
            - Include expert quotes
            - Maintain factual accuracy
         4. Digital Optimization 💻
            - Structure for scanability
            - Include shareable takeaways
            - Optimize for SEO
            - Add engaging subheadings\
         """),
            expected_output=dedent("""\
         # {Viral-Worthy Headline}

         ## Introduction
         {Engaging hook and context}

         ## {Compelling Section 1}
         {Key insights and analysis}
         {Expert quotes and statistics}

         ## {Engaging Section 2}
         {Deeper exploration}
         {Real-world examples}

         ## {Practical Section 3}
         {Actionable insights}
         {Expert recommendations}

         ## Key Takeaways
         - {Shareable insight 1}
         - {Practical takeaway 2}
         - {Notable finding 3}

         ## Sources
         {Properly attributed sources with links}\
         """),
            markdown=True,
        )
