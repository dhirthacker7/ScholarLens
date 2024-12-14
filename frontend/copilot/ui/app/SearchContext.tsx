import React, { createContext, useContext, useState, ReactNode } from "react";

// Define the context type
interface SearchContextType {
  results: any[];
  setResults: React.Dispatch<React.SetStateAction<any[]>>;
}

// Create the context
const SearchContext = createContext<SearchContextType | undefined>(undefined);

// Custom hook for using the context
export const useSearchContext = () => {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error("useSearchContext must be used within a SearchProvider");
  }
  return context;
};

// Context provider component
export function SearchProvider({ children }: { children: ReactNode }) {
  const [results, setResults] = useState<any[]>([]); // Initialized as an empty array

  return (
    <SearchContext.Provider value={{ results, setResults }}>
      {children}
    </SearchContext.Provider>
  );
}
