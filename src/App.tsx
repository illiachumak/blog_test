import React from 'react';

function App() {
  return (
    <>
      <header className="header">
        <h1>React Blog</h1>
      </header>
      <main>
        <section className='search'>
          <form action="">
            <input type="text" />
          </form>
          <div className='search-results'></div>
        </section>
        <section className='article-wrap'>
          <article>

          </article>
        </section>
        <section className='pagination'>

        </section>
      </main>
    </>
  );
}

export default App;
