from yaml import load
from yaml import Loader

import aiohttp
import asyncio
import uvloop

from bs4 import BeautifulSoup


class ArticleLoader:
    def __init__(self, path):
        with open(path) as fd:
            text = fd.read()

        self.articles = load(text, Loader=Loader)

    def load(self):
        uvloop.install()
        asyncio.run(self._load())

        return [article['text'] for article in self.articles]

    async def _load(self):
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[self.fetch(session, article) for article in self.articles])
    
    async def fetch(self, session, article: dict):
        async with session.get(article['url']) as resp:
            html_content = await resp.text()
            article['text'] = self._parser(html_content, **article)

    @staticmethod
    def _parser(html_content, selectors, content_type='text', sep='\n', **kwargs):
        soup = BeautifulSoup(html_content, 'lxml')

        if content_type == 'text':
            result = sep.join(
                ''.join(element.get_text() for element in soup.select(sel)) 
                for sel in selectors) 
        else:
            raise NotImplementedError(f'{content_type} Not Implemented')

        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="A configurable crawler that reads on a YAML file")

    parser.add_argument('file', help="file path")
    parser.add_argument('--sep', help="strings between articles", dest="sep",
                        default="\n\n===================================\n\n")

    args = parser.parse_args()

    article_loader = ArticleLoader(args.file)
    article_loader.load()

    print(*[article['text'] for article in article_loader.articles], sep=args.sep)
