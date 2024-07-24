import React, { useState } from 'react';
import { Article } from '../services/types.type';

interface SearchProps{
    articles: Article[];
}

const SearchComponent = ({articles}: SearchProps) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [suggestions, setSuggestions] = useState<Article[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    setSearchTerm(event.target.value);

    const filteredSuggestions = articles.filter(suggestion =>
      suggestion.title.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setSuggestions(filteredSuggestions);
    setShowSuggestions(filteredSuggestions.length > 0);
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    alert(`You searched for: ${searchTerm}`);
    setSearchTerm('');
    setSuggestions([]);
    setShowSuggestions(false);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={searchTerm}
        onChange={handleInputChange}
        placeholder="Search..."
      />
      {showSuggestions && searchTerm.trim().length ? (
      <div className='search-results'>
        <ul>
          {suggestions.map((suggestion, i) => {
            if(i>5) return
            return(
            <li key={i}>{suggestion.title}</li>
          )})}
        </ul>
      </div>
      ) : <></>}
      <button type="submit">Search</button>
    </form>
  );
};

export default SearchComponent;
