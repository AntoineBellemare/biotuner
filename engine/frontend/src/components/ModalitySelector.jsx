import { Music, Heart, Brain, Smartphone, Leaf, Sparkles, Waves } from 'lucide-react'

const modalityInfo = {
  audio: {
    title: '🎵 Audio Signals',
    description: 'Audio signals are everywhere. Try to find the tuning of a bell, the resonance of a yawn, or the subharmonies of a whale song. Transform the takeoff of a plane into a chord, or extract the hidden harmonies of the river where you swim.',
    resources: [
      { name: 'Sonic Visualiser', url: 'https://www.sonicvisualiser.org/', desc: 'Advanced tool for visualizing and analyzing audio features.' },
      { name: 'Freesound', url: 'https://freesound.org/', desc: 'A collaborative database of audio samples, field recordings, and sound effects.' },
      { name: 'Librosa', url: 'https://librosa.org/', desc: 'Python library for audio analysis and feature extraction.' },
    ]
  },
  brain: {
    title: '🧠 Brain Signals (EEG)',
    description: 'EEG captures electrical activity from populations of neurons in the brain, measured via electrodes on the scalp. It provides insights into neural oscillations, event-related potentials (ERP), and functional connectivity.',
    resources: [
      { name: 'OpenNeuro', url: 'https://openneuro.org/', desc: 'A large repository of open EEG, fMRI, and MEG datasets for neuroscience research.' },
      { name: 'BNCI Horizon 2020', url: 'http://bnci-horizon-2020.eu/database', desc: 'A collection of EEG datasets related to brain-computer interfaces (BCI) and cognitive experiments.' },
      { name: 'OpenBCI', url: 'https://openbci.com/', desc: 'Open-source EEG hardware and software platform for neurophysiology research and BCI applications.' },
    ]
  },
  heart: {
    title: '❤️ Heart Signals (ECG)',
    description: 'ECG measures the heart\'s electrical activity, revealing information on cardiac cycles, heart rate variability (HRV), and arrhythmias. Got a smartwatch? Use your own heart rate data to explore the rhythm and music of your pulses.',
    resources: [
      { name: 'PhysioNet', url: 'https://physionet.org/', desc: 'Open database of ECG recordings.' },
      { name: 'Neurokit2', url: 'https://neuropsychology.github.io/NeuroKit/', desc: 'A Python toolbox for physiological signal processing.' },
    ]
  },
  sensors: {
    title: '📱 Smartphone Sensors',
    description: 'Smartphones integrate multiple sensors including accelerometers (motion), gyroscopes (rotation), magnetometers (orientation), and PPG (heart rate). These sensors enable multimodal physiological and behavioral tracking.',
    resources: [
      { name: 'Physics Toolbox Sensor Suite', url: 'https://www.vieyrasoftware.net/', desc: 'Logs multiple smartphone sensor streams.' },
      { name: 'TouchOSC', url: 'https://hexler.net/touchosc', desc: 'A powerful mobile app for sending sensor data via OSC, widely used for real-time interactive applications.' },
      { name: 'Phyphox', url: 'https://phyphox.org/', desc: 'Open-source app for experimental sensor data collection.' },
    ]
  },
  plant: {
    title: '🌿 Plant Signals',
    description: 'Plants generate bioelectrical signals influenced by external stimuli such as light, touch, and environmental changes. These signals can be measured as action potentials, impedance changes, or ion channel activity. Let\'s shift our timeframes to listen to the music of the plants.',
    resources: [
      { name: 'Plant SpikerBox', url: 'https://backyardbrains.com/products/PlantSpikerBox', desc: 'DIY plant electrophysiology kit.' },
    ]
  },
  creative: {
    title: '✨ Your Creativity',
    description: 'Any data that fluctuates over time can become a sound, a visualization, or an insight. Try transforming stock market data, dance movements, or even gravitational waves into harmonies.',
    resources: []
  },
}

export default function ModalitySelector({ selected, onSelect }) {
  const modalities = [
    { 
      id: 'audio', 
      name: 'Audio', 
      icon: Music, 
      color: 'from-biotuner-primary to-cyan-400',
      emoji: '🎵'
    },
    { 
      id: 'brain', 
      name: 'Brain (EEG)', 
      icon: Brain, 
      color: 'from-biotuner-secondary to-purple-500',
      emoji: '🧠'
    },
    { 
      id: 'heart', 
      name: 'Heart (ECG)', 
      icon: Heart, 
      color: 'from-red-500 to-pink-500',
      emoji: '❤️'
    },
    { 
      id: 'sensors', 
      name: 'Smartphone Sensors', 
      icon: Smartphone, 
      color: 'from-biotuner-accent to-emerald-500',
      emoji: '📱'
    },
    { 
      id: 'plant', 
      name: 'Plant', 
      icon: Leaf, 
      color: 'from-green-500 to-lime-500',
      emoji: '🌿'
    },
    { 
      id: 'creative', 
      name: 'Your Creativity', 
      icon: Sparkles, 
      color: 'from-yellow-500 to-amber-500',
      emoji: '✨'
    },
  ]

  return (
    <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-8">
      <h2 className="text-xs font-semibold mb-6 text-biotuner-light/60 uppercase tracking-wider flex items-center gap-2">
        <Waves className="w-4 h-4" />
        Supported Modalities
      </h2>
      
      <div className="grid grid-cols-3 gap-4">
        {modalities.map((modality) => {
          const Icon = modality.icon
          return (
            <button
              key={modality.id}
              onClick={() => onSelect(modality.id)}
              className={`
                relative group p-4 rounded-lg border-2 transition-all duration-300 overflow-hidden
                ${selected === modality.id
                  ? 'border-biotuner-primary bg-biotuner-primary/10'
                  : 'border-biotuner-dark-600 bg-biotuner-dark-800/50 hover:border-biotuner-primary/50'
                }
              `}
            >
              <div className="relative z-10 text-center">
                <div className="text-3xl mb-2">{modality.emoji}</div>
                <div className="text-sm font-medium text-biotuner-light">{modality.name}</div>
              </div>
              
              {selected === modality.id && (
                <div className={`absolute inset-0 bg-gradient-to-br ${modality.color} opacity-5`}></div>
              )}
            </button>
          )
        })}
      </div>
      
      {/* Info Panel */}
      {selected && modalityInfo[selected] && (
        <div className="mt-6 p-6 bg-biotuner-dark-800/70 rounded-lg border-l-4 border-biotuner-primary">
          <h3 className="text-lg font-bold text-biotuner-light mb-3">
            {modalityInfo[selected].title}
          </h3>
          
          <p className="text-biotuner-light/80 text-sm leading-relaxed mb-4">
            {modalityInfo[selected].description}
          </p>
          
          {modalityInfo[selected].resources.length > 0 && (
            <>
              <h4 className="text-xs font-semibold text-biotuner-light/60 uppercase tracking-wider mb-3">
                Resources:
              </h4>
              <div className="space-y-2">
                {modalityInfo[selected].resources.map((resource, idx) => (
                  <div key={idx} className="flex items-start gap-2">
                    <span className="text-biotuner-primary mt-0.5">🔹</span>
                    <div className="flex-1">
                      <a
                        href={resource.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-biotuner-primary hover:text-biotuner-primary/80 font-medium text-sm transition-colors underline"
                      >
                        {resource.name}
                      </a>
                      <span className="text-biotuner-light/60 text-sm"> – {resource.desc}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
