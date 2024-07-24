import { useEffect, useState } from 'react';
import { fetchArticles } from './services/api';
import { Article } from './services/types.type';
import ArticlesList from './components/ArticlesList';
import Pagination from './components/Pagination';
import { defaultPageSize } from './services/constants';
import { useSearchParams } from 'react-router-dom';
import SearchComponent from './components/Search';

function App() {

  const [articles, setArticles] = useState<Article[]>([])
  const [articlesToRender, setArticlesToRender] = useState<Article[]>([])
  const [currentPage, setCurrentPage] = useState(1)
  const [maxPages, setMaxPages] = useState(1)
  const [searchParams, setSearchParams] = useSearchParams()

  useEffect(()=>{
    const currentPage = Number(searchParams.get('page')) || 1;
    setCurrentPage(currentPage);
    (async () => {
      const articles = await fetchArticles()
      if(articles.length){
        setMaxPages(articles.length/defaultPageSize)
        const start = (currentPage - 1) * defaultPageSize;
        const end = Math.min(currentPage * defaultPageSize, articles.length)
        const articlesToRender = [...articles]
        setArticles(articles)
        setArticlesToRender(articlesToRender.slice(start, end))
      }
    })()
  },[searchParams])

  return (
    <>
      <header className="header container">
        <h1>React Blog</h1>
      </header>
      <main className='container'>
        <section className='search'>
          <SearchComponent articles={articles}/>
        </section>
        <section className='articles-wrap'>
          {articlesToRender.length && <ArticlesList articles={articlesToRender}/>}
        </section>
        <section className='pagination'>
          <Pagination
            currentPage={currentPage}
            totalPages={maxPages}
          />
        </section>
      </main>
    </>
  );
}

export default App;
