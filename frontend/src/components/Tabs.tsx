import { useState } from 'react';

interface TabsProps {
  children: React.ReactNode[];
  labels: string[];
}

export function Tabs({ children, labels }: TabsProps) {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div>
      <div className="flex space-x-4 mb-6 border-b border-gray-700">
        {labels.map((label, index) => (
          <button
            key={label}
            className={`px-4 py-2 -mb-px ${
              activeTab === index
                ? 'border-b-2 border-blue-500 text-blue-500'
                : 'text-gray-400 hover:text-gray-300'
            }`}
            onClick={() => setActiveTab(index)}
          >
            {label}
          </button>
        ))}
      </div>
      {children[activeTab]}
    </div>
  );
} 