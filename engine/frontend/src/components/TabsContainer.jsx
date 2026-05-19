import { useState } from 'react'
import TuningTab from './tabs/TuningTab'
import ChordsTab from './tabs/ChordsTab'
import BiocolorsTab from './tabs/BiocolorsTab'
import GuitarTuningTab from './tabs/GuitarTuningTab'

export default function TabsContainer({ sessionId, analysisResult, analysisConfig, fileInfo, onSaveTuning }) {
  const [activeTab, setActiveTab] = useState('tuning')

  const tabs = [
    { id: 'tuning', label: 'Tuning', icon: '🎼' },
    { id: 'guitar', label: 'Guitar', icon: '🎸' },
    { id: 'chords', label: 'Chords', icon: '🎹' },
    { id: 'biocolors', label: 'Biocolors', icon: '🎨' },
  ]

  return (
    <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 overflow-hidden">
      {/* Tab Headers - Scrollable on mobile */}
      <div className="flex border-b border-biotuner-dark-600 bg-biotuner-dark-900 overflow-x-auto scrollbar-hide">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              relative flex-1 min-w-[100px] px-4 sm:px-8 py-3 sm:py-4 text-xs sm:text-sm font-medium uppercase tracking-wider transition-all duration-300 whitespace-nowrap
              ${activeTab === tab.id
                ? 'text-biotuner-primary'
                : 'text-biotuner-light/40 hover:text-biotuner-light/80'
              }
            `}
          >
            <span className="relative z-10 flex items-center justify-center gap-1 sm:gap-2">
              <span>{tab.icon}</span>
              <span className="hidden xs:inline">{tab.label}</span>
            </span>
            {activeTab === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-biotuner-primary to-biotuner-secondary"></div>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-4 sm:p-6 lg:p-8 bg-biotuner-dark-800">
        {activeTab === 'tuning' && (
          <TuningTab
            sessionId={sessionId}
            analysisResult={analysisResult}
            fileInfo={fileInfo}
          />
        )}
        {activeTab === 'guitar' && (
          <GuitarTuningTab
            analysisResult={analysisResult}
            onSaveToLibrary={onSaveTuning}
          />
        )}
        {activeTab === 'chords' && (
          <ChordsTab
            sessionId={sessionId}
            analysisResult={analysisResult}
            analysisConfig={analysisConfig}
            fileInfo={fileInfo}
          />
        )}
        {activeTab === 'biocolors' && (
          <BiocolorsTab
            sessionId={sessionId}
            analysisResult={analysisResult}
            fileInfo={fileInfo}
          />
        )}
      </div>
    </div>
  )
}
