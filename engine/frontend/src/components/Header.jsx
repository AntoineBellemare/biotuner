import { Menu } from 'lucide-react'

export default function Header({ onMenuToggle }) {
  return (
    <header className="border-b border-biotuner-dark-600 bg-biotuner-dark-900/50 backdrop-blur-xl">
      <div className="px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
        <div className="flex items-center justify-between gap-4">
          {/* Mobile menu button */}
          <button
            onClick={onMenuToggle}
            className="lg:hidden p-2 rounded-lg bg-biotuner-dark-800 hover:bg-biotuner-dark-700 transition-colors"
            aria-label="Toggle menu"
          >
            <Menu className="w-6 h-6 text-biotuner-primary" />
          </button>
          
          <div className="flex-1 min-w-0">
            <h1 className="text-2xl sm:text-3xl lg:text-5xl font-bold tracking-wider bg-gradient-to-r from-biotuner-primary via-biotuner-secondary to-biotuner-accent bg-clip-text text-transparent mb-1 sm:mb-2 truncate">
              BIOTUNER ENGINE
            </h1>
            <p className="text-xs sm:text-sm tracking-widest text-biotuner-light/60 uppercase hidden sm:block">
              Harmonic Analysis of Time Series
            </p>
          </div>
          <div className="flex items-center gap-3 sm:gap-6 flex-shrink-0">
            {/* Kairos~Hive Branding */}
            <a 
              href="https://kairos-hive.org" 
              target="_blank" 
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-3 px-3 sm:px-4 py-2 rounded-lg bg-gradient-to-r from-purple-900/30 to-indigo-900/30 border border-purple-500/30 hover:border-purple-400/50 transition-all duration-300 group hover:shadow-lg hover:shadow-purple-500/20"
            >
              <img 
                src="/kairos-logo.webp" 
                alt="Kairos~Hive" 
                className="h-6 w-6 sm:h-8 sm:w-8 object-contain group-hover:scale-110 transition-transform duration-300"
              />
              <div className="flex flex-col">
                <span className="text-xs text-gray-400 uppercase tracking-wider">Powered by</span>
                <span className="text-sm font-semibold bg-gradient-to-r from-purple-400 to-indigo-400 bg-clip-text text-transparent group-hover:from-purple-300 group-hover:to-indigo-300">
                  Kairos~Hive
                </span>
              </div>
            </a>
            
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-biotuner-accent rounded-full animate-pulse"></div>
              <span className="text-xs sm:text-sm text-biotuner-light/60 hidden xs:inline">Connected</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
