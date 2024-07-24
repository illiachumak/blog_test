import { defaultPageSize } from "../services/constants";
import { Article } from "../services/types.type"

export interface ArticlesListProps{
    articles: Article[]
}

const ArticlesList = ({articles}: ArticlesListProps) => {
    return (
        <>
        {articles.length && articles.map((item, i) => {
             return(
              <article key={i}>
                <h2>{item?.title}</h2>
                <p>{item?.short_description}</p>
              </article>
            )
          })}
        </>
        )

}

export default ArticlesList
