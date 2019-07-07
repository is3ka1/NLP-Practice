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
    article_loader = ArticleLoader('target.yaml')
    article_loader.load()
    for article in article_loader.articles:
        print(article['text'], end='\n\n\n')
