#!/usr/bin/env python3
"""
update_ai_top_papers.py
=======================

This script retrieves recent research announcements and blog posts from several
prominent AI-focused sources and assembles a JSON feed containing the top
items. The current sources include the Hugging Face blog, the OpenAI blog,
the MIT News artificial intelligence channel, and the arXiv categories for
Artificial Intelligence (cs.AI) and Machine Learning (cs.LG). The feed is
sorted by publication date so that the most recent items appear first, and
only the most recent ten entries are kept.

Each entry in the resulting feed contains the following fields:

* ``title`` – The title of the post or paper.
* ``source`` – A short identifier describing where the entry originated.
* ``url`` – A direct link to the item.
* ``published`` – The publication date formatted in ISO-8601 (or ``null`` if
  unavailable).
* ``summary`` – A brief summary extracted from the description field in the
  feed. The summary is truncated to the first two sentences to remain
  concise.

To execute this script you need an internet connection. No third-party
packages are required for the basic feed generation; everything is
implemented with Python’s standard library. If you supply an OpenAI API key
and e-mail configuration via environment variables, the script will also
summarise each entry into an SEO-friendly article and send a daily digest to
a list of subscribers.

"""

import json
import datetime
import html
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET


def parse_rss_feed(url: str, source: str) -> list[dict]:
    """Fetch and parse an RSS or Atom feed.

    Args:
        url: URL of the RSS feed to parse.
        source: A string identifier for the feed (e.g., ``"huggingface"``).

    Returns:
        A list of dictionaries representing feed items. Each dictionary has
        ``title``, ``url``, ``published`` (a datetime object or ``None``)
        and ``summary``.

    Raises:
        urllib.error.URLError: If there is an issue connecting to the feed.
        xml.etree.ElementTree.ParseError: If the feed cannot be parsed.
    """
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch feed from {url}: {exc}") from exc

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse feed XML from {url}: {exc}") from exc

    items: list[dict] = []
    # Items are usually under <item> for RSS or <entry> for Atom
    for item in root.iter():
        tag = item.tag.lower()
        if tag.endswith('item') or tag.endswith('entry'):
            title_node = item.find('title')
            link_node = item.find('link')
            # Atom feeds often provide <link href="..." /> instead of text
            link_href = None
            if link_node is not None:
                if link_node.text and link_node.text.strip():
                    link_href = link_node.text.strip()
                else:
                    link_href = link_node.get('href')

            pub_date_node = (
                item.find('pubDate') or item.find('published') or
                item.find('{http://www.w3.org/2005/Atom}published') or
                item.find('updated')
            )
            summary_node = (
                item.find('description') or item.find('summary') or
                item.find('{http://www.w3.org/2005/Atom}summary')
            )
            # Extract text values
            title = html.unescape(title_node.text.strip()) if title_node is not None and title_node.text else ''
            url_link = link_href.strip() if link_href else ''
            raw_summary = html.unescape(summary_node.text.strip()) if summary_node is not None and summary_node.text else ''

            # Parse publication date; fallback to None if unparsable
            published_dt: datetime.datetime | None = None
            if pub_date_node is not None and pub_date_node.text:
                date_text = pub_date_node.text.strip()
                # Try common date formats
                for fmt in (
                    '%a, %d %b %Y %H:%M:%S %z',  # RFC 822 (e.g., Tue, 20 Apr 2021 15:30:00 +0000)
                    '%Y-%m-%dT%H:%M:%SZ',         # ISO 8601 with Z
                    '%Y-%m-%dT%H:%M:%S%z',         # ISO 8601 with offset
                    '%Y-%m-%d',                    # Date only
                ):
                    try:
                        published_dt = datetime.datetime.strptime(date_text, fmt)
                        break
                    except ValueError:
                        continue
                # If still none, attempt to parse via email.utils (slow path)
                if published_dt is None:
                    try:
                        from email.utils import parsedate_to_datetime  # type: ignore
                        published_dt = parsedate_to_datetime(date_text)
                    except Exception:
                        published_dt = None

            # Shorten summary: use only the first two sentences or first 40 words
            summary_clean = raw_summary.replace('\n', ' ').strip()
            summary_sentence_split = summary_clean.split('. ')
            if summary_sentence_split:
                truncated = '. '.join(summary_sentence_split[:2])
                if truncated and not truncated.endswith('.'):
                    truncated += '.'
            else:
                # Fallback to a word limit
                words = summary_clean.split()[:40]
                truncated = ' '.join(words)
                if truncated and not truncated.endswith('.'):
                    truncated += '...'

            items.append({
                'title': title,
                'source': source,
                'url': url_link,
                'published': published_dt,
                'summary': truncated,
            })

    return items


def build_feed() -> list[dict]:
    """Construct a combined list of feed items from all configured sources."""
    feeds = {
        'huggingface': 'https://huggingface.co/blog/rss.xml',
        'openai': 'https://openai.com/blog/rss',
        'mit': 'https://news.mit.edu/topic/artificial-intelligence/rss',
        'arxiv_cs_ai': 'http://export.arxiv.org/rss/cs.AI',
        'arxiv_cs_lg': 'http://export.arxiv.org/rss/cs.LG',
    }
    combined_items: list[dict] = []
    for source_name, feed_url in feeds.items():
        try:
            items = parse_rss_feed(feed_url, source_name)
            combined_items.extend(items)
        except Exception as exc:
            # In production you might want to log this error; for now we just
            # print to stderr and continue.
            print(f"Warning: failed to load feed '{source_name}': {exc}", file=sys.stderr)
    # Sort by publication date descending; None dates go last
    combined_items.sort(key=lambda item: item['published'] or datetime.datetime.min, reverse=True)
    # Keep only the most recent ten entries
    top_items = combined_items[:10]
    # Convert datetime objects to ISO strings for JSON serialization
    for entry in top_items:
        dt = entry['published']
        entry['published'] = dt.isoformat() if isinstance(dt, datetime.datetime) else None
    return top_items


def generate_seo_article(title: str, summary: str, url: str) -> str:
    """
    Generate an SEO-friendly article using OpenAI or fall back to the original summary.

    This function uses the OpenAI API if an API key is available via the
    ``OPENAI_API_KEY`` environment variable. It composes a prompt that
    instructs the model to produce a concise, keyword-rich article based on
    the provided title and summary. If the API is unavailable or an error
    occurs, it returns a simple fallback message with a link to the source.
    """
    import os
    try:
        import openai  # type: ignore
    except Exception:
        openai = None  # type: ignore

    if openai is None or not os.getenv('OPENAI_API_KEY'):
        return f"{title}\n\n{summary}\n\nRead more: {url}"

    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = (
        "You are a helpful assistant who writes concise, SEO-friendly news articles. "
        "Use the following information about an AI news item to craft a 150–200-word "
        "article that highlights the significance of the work and includes relevant keywords "
        "like AI, machine learning and deep learning. Include the title at the beginning "
        "as a heading and end with a call to action linking to the original source.\n\n"
        f"Title: {title}\nSummary: {summary}\nURL: {url}\n"
    )
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=350,
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as exc:
        return f"{title}\n\n{summary}\n\nRead more: {url}\n\n(Note: AI summary unavailable: {exc})"


def send_email(subject: str, body: str) -> None:
    """
    Send the assembled AI summaries via e-mail to subscribers.

    This function relies on SMTP credentials and a comma-separated list of
    subscriber addresses provided via the ``EMAIL_HOST``, ``EMAIL_PORT``,
    ``EMAIL_USERNAME``, ``EMAIL_PASSWORD``, ``EMAIL_SENDER`` and
    ``SUBSCRIBERS`` environment variables. If any of these values are
    missing, the e-mail sending step is skipped.
    """
    import os
    import sys
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    host = os.getenv('EMAIL_HOST')
    port = int(os.getenv('EMAIL_PORT', '465'))
    username = os.getenv('EMAIL_USERNAME')
    password = os.getenv('EMAIL_PASSWORD')
    sender = os.getenv('EMAIL_SENDER', username)
    subscribers = os.getenv('SUBSCRIBERS')

    if not (host and username and password and subscribers):
        print("Email configuration or subscriber list missing; skipping email.", file=sys.stderr)
        return

    recipients = [addr.strip() for addr in subscribers.split(',') if addr.strip()]
    if not recipients:
        print("No valid subscribers found; skipping email.", file=sys.stderr)
        return

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        with smtplib.SMTP_SSL(host, port) as server:
            server.login(username, password)
            server.sendmail(sender, recipients, msg.as_string())
    except Exception as exc:
        print(f"Failed to send email: {exc}", file=sys.stderr)


def main(path: str = 'ai_top_papers.json') -> None:
    """Main entry point: build the feed, write JSON and send summaries."""
    top_feed = build_feed()
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(top_feed, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"Error writing JSON feed to {path}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Generate SEO articles and email them
    articles = [generate_seo_article(item['title'], item['summary'], item['url']) for item in top_feed]
    if articles:
        subject = "Daily AI News Digest"
        body = "\n\n".join(articles)
        send_email(subject, body)


if __name__ == '__main__':
    outfile = 'ai_top_papers.json'
    if len(sys.argv) > 1:
        outfile = sys.argv[1]
    main(outfile)
