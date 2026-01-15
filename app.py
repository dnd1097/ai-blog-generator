import re
import tempfile

import gradio as gr
import markdown

from ai_blog_generator.agents import BlogAgents
from ai_blog_generator.generator import BlogPostGenerator
from ai_blog_generator.model import Model
from ai_blog_generator.utils import custom_css, get_default_llm


def generate_blog(llm_provider, llm_name, api_key, user_topic, style_guidelines, include_sources):
    if not api_key:
        gr.Warning(f"Please enter your {llm_provider} API key.")
    if not llm_name or llm_name.strip() == "":
        gr.Warning("Please enter a model name.")
    if not user_topic or user_topic.strip() == "":
        gr.Warning("Please enter a blog topic or ideas for the topic.")
        return gr.update(value="", visible=True), "", gr.update(value=None, visible=False)
    url_safe_topic = re.sub(r"\s+", "-", user_topic.strip().lower())
    llm = Model(llm_provider, llm_name, api_key)
    blog_agents = BlogAgents(llm)
    generate_blog_post = BlogPostGenerator(
        blog_agents=blog_agents,
        session_id=f"generate-blog-post-on-{url_safe_topic}",
        debug_mode=True,
    )
    blog_post = generate_blog_post.run(
        topic=user_topic,
        style_guidelines=style_guidelines,
        include_sources=include_sources,
    )
    final_output = ""
    sources = set()
    for response in blog_post:
        if hasattr(response, "content") and response.content:
            final_output += str(response.content) + "\n"
        if hasattr(response, "sources") and response.sources:
            if isinstance(response.sources, (list, set)):
                sources.update(response.sources)
            else:
                sources.add(str(response.sources))

    markdown_output = final_output
    if include_sources and sources:
        markdown_output += "\n\n## Sources\n"
        markdown_output += "\n".join(f"- {src}" for src in sorted(sources))
    html_body = markdown.markdown(markdown_output)
    html_content = f"<div>{html_body}</div>"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(markdown_output)
        download_path = temp_file.name
    return (
        gr.update(value=html_content, visible=True),
        "",
        gr.update(value=download_path, visible=True),
    )


with gr.Blocks(title="Blog Generator", css=custom_css) as demo:
    gr.Markdown("# AI Blog Generator", elem_classes="center-text")
    with gr.Row():
        with gr.Column(scale=1):
            llm_provider = gr.Radio(
                label="Select LLM Provider",
                choices=["OpenAI", "Gemini", "Claude", "Grok"],
                value="Gemini",
            )

            # Function to update the textbox when provider changes
            def update_llm_name(provider):
                return get_default_llm(provider)

            llm_name = gr.Textbox(
                label="Enter LLM Name",
                value=get_default_llm(llm_provider.value),
                info="Specify the model name based on the provider.",
            )
            # When provider changes, update the textbox
            llm_provider.change(fn=update_llm_name, inputs=llm_provider, outputs=llm_name)

            api_key = gr.Textbox(label="Enter API Key", type="password")
            user_topic = gr.Textbox(
                label="Enter your own blog topic",
                lines=4,
                placeholder="Share your ideas, angle, or key points for the blog topic.",
            )
            style_guidelines = gr.Textbox(
                label="Style guidelines",
                lines=4,
                placeholder="Optional: tone, length, target audience, medium, SEO needs, etc.",
            )
            include_sources = gr.Checkbox(label="Include sources section", value=False)
            generate_btn = gr.Button("Generate Blog")
        with gr.Column(scale=2):
            with gr.Row(elem_classes="top-right"):
                download_btn = gr.DownloadButton(
                    label="Download Markdown",
                    visible=False,
                )
            output = gr.HTML(
                label="Generated Post",
                visible=True,
            )
            warning = gr.Textbox(label="Warning", visible=False)

    generate_btn.click(
        generate_blog,
        inputs=[llm_provider, llm_name, api_key, user_topic, style_guidelines, include_sources],
        outputs=[output, warning, download_btn],
    )

demo.launch()
