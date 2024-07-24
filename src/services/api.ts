import { baseApiUrl } from "./constants"
import { Article } from "./types.type";

export const fetchArticles = async (): Promise<Article[]> => {
    const response = await fetch(baseApiUrl + '/posts');
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const articles: Article[] = await response.json();
    return articles;
  };
