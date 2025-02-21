export function InfoIcon({ className = "w-5 h-5" }: { className?: string }) {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      className={className}
    >
      <circle cx="12" cy="12" r="10" strokeWidth="2"/>
      <line x1="12" y1="16" x2="12" y2="12" strokeWidth="2"/>
      <line x1="12" y1="8" x2="12" y2="8" strokeWidth="2"/>
    </svg>
  );
} 