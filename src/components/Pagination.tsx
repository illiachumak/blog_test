import React from 'react';
import { Link } from 'react-router-dom';
import { useSearchParams } from 'react-router-dom';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
}

function Pagination({ currentPage, totalPages }: PaginationProps) {
    const [searchParams, setSearchParams] = useSearchParams()

    const handlePageChange = (page: any) => {
        setSearchParams({ page });
    };

    const pages: React.ReactElement[] = [];
    for (let i = 1; i <= totalPages; i++) {
        pages.push(
        <Link
            key={i}
            to={"?page=" + i}
            className={`pagination-item ${currentPage === i ? 'active' : ''}`}
            onClick={() => handlePageChange(i)}
        >
            {i}
        </Link>
        );
    }

  return (
    <section className='pagination'>
      <Link
        to="?page=1"
        className={`pagination-item ${currentPage === 1 ? 'disabled' : ''}`}
        onClick={() => handlePageChange(1)}
      >
        &laquo;
      </Link>
      {pages}
      <Link
        to={"?page=" + totalPages}
        className={`pagination-item ${currentPage === totalPages ? 'disabled' : ''}`}
        onClick={() => handlePageChange(totalPages)}
      >
        &raquo;
      </Link>
    </section>
  );
}

export default Pagination;