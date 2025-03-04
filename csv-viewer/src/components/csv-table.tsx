// src/components/CsvTable.tsx
import { FC, useState, useMemo } from 'react';
import { CsvData } from '../types';

interface CsvTableProps {
  data: CsvData[];
}

const CsvTable: FC<CsvTableProps> = ({ data }) => {
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 15;

  if (!data.length) return <div className="text-center py-10">No data available</div>;

  const headers = Object.keys(data[0]);

  // Handle sorting
  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  // Filter and sort data
  const filteredAndSortedData = useMemo(() => {
    let processedData = [...data];

    // Filter data
    if (searchTerm) {
      processedData = processedData.filter(row => 
        Object.values(row).some(value => 
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    // Sort data
    if (sortColumn) {
      processedData.sort((a, b) => {
        const aValue = a[sortColumn];
        const bValue = b[sortColumn];

        // Check if values are numbers
        const aNum = Number(aValue);
        const bNum = Number(bValue);

        if (!isNaN(aNum) && !isNaN(bNum)) {
          return sortDirection === 'asc' ? aNum - bNum : bNum - aNum;
        }

        // String comparison
        const aStr = String(aValue).toLowerCase();
        const bStr = String(bValue).toLowerCase();
        
        if (sortDirection === 'asc') {
          return aStr.localeCompare(bStr);
        } else {
          return bStr.localeCompare(aStr);
        }
      });
    }

    return processedData;
  }, [data, searchTerm, sortColumn, sortDirection]);

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedData.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedData = filteredAndSortedData.slice(startIndex, startIndex + rowsPerPage);

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Search and pagination controls */}
      <div className="p-4 bg-gray-50 border-b flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="w-full sm:w-64">
          <input
            type="text"
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1); // Reset to first page on search
            }}
            className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">
            Showing {filteredAndSortedData.length > 0 ? startIndex + 1 : 0} - {Math.min(startIndex + rowsPerPage, filteredAndSortedData.length)} of {filteredAndSortedData.length}
          </span>
          <div className="flex space-x-1">
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 rounded border bg-white disabled:opacity-50"
            >
              &lt;
            </button>
            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages || totalPages === 0}
              className="px-3 py-1 rounded border bg-white disabled:opacity-50"
            >
              &gt;
            </button>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-100">
            <tr>
              {headers.map((header) => (
                <th 
                  key={header} 
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-200"
                  onClick={() => handleSort(header)}
                >
                  <div className="flex items-center">
                    {header.replace(/_/g, ' ')}
                    {sortColumn === header && (
                      <span className="ml-1">
                        {sortDirection === 'asc' ? '▲' : '▼'}
                      </span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {paginatedData.map((row, rowIndex) => (
              <tr 
                key={rowIndex} 
                className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
              >
                {headers.map((header, cellIndex) => {
                  // Format cell based on content
                  const cellValue = row[header];
                  const isNumber = !isNaN(Number(cellValue)) && cellValue !== '';
                  
                  return (
                    <td 
                      key={`${rowIndex}-${cellIndex}`} 
                      className={`px-6 py-4 whitespace-nowrap text-sm ${
                        isNumber ? 'text-right font-mono' : 'text-left'
                      } ${
                        // Highlight positive/negative values
                        isNumber && Number(cellValue) > 0 ? 'text-green-600' :
                        isNumber && Number(cellValue) < 0 ? 'text-red-600' : 'text-gray-700'
                      }`}
                    >
                      {cellValue}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination footer */}
      {totalPages > 1 && (
        <div className="bg-gray-50 px-4 py-3 flex items-center justify-between border-t">
          <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
            <div>
              <p className="text-sm text-gray-700">
                Page <span className="font-medium">{currentPage}</span> of{' '}
                <span className="font-medium">{totalPages}</span>
              </p>
            </div>
            <div>
              <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
                <button
                  onClick={() => setCurrentPage(1)}
                  disabled={currentPage === 1}
                  className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                >
                  First
                </button>
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  // Show pages around current page
                  let pageNum;
                  if (totalPages <= 5) {
                    pageNum = i + 1;
                  } else if (currentPage <= 3) {
                    pageNum = i + 1;
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i;
                  } else {
                    pageNum = currentPage - 2 + i;
                  }
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium
                        ${currentPage === pageNum 
                          ? 'z-10 bg-blue-50 border-blue-500 text-blue-600' 
                          : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                        }`}
                    >
                      {pageNum}
                    </button>
                  );
                })}
                <button
                  onClick={() => setCurrentPage(totalPages)}
                  disabled={currentPage === totalPages}
                  className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                >
                  Last
                </button>
              </nav>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CsvTable;