import { useMemo } from 'react'

// Convert decimal to fraction using max denominator
function decimalToFraction(decimal, maxDenom = 100) {
  let bestNum = 1
  let bestDen = 1
  let minError = Math.abs(decimal - 1)
  
  for (let den = 1; den <= maxDenom; den++) {
    const num = Math.round(decimal * den)
    const error = Math.abs(decimal - num / den)
    if (error < minError) {
      minError = error
      bestNum = num
      bestDen = den
    }
  }
  
  // Reduce fraction
  const gcd = (a, b) => b === 0 ? a : gcd(b, a % b)
  const divisor = gcd(bestNum, bestDen)
  return `${bestNum / divisor}/${bestDen / divisor}`
}

// Magma colormap interpolation (from matplotlib)
function magmaColor(t) {
  // t should be between 0 and 1
  t = Math.max(0, Math.min(1, t))
  
  // Magma colormap key points (approximation)
  const magmaColors = [
    [0.001462, 0.000466, 0.013866],
    [0.087411, 0.044556, 0.224813],
    [0.258234, 0.038571, 0.406485],
    [0.417642, 0.090203, 0.432943],
    [0.596422, 0.182494, 0.415088],
    [0.762373, 0.334290, 0.335347],
    [0.900704, 0.543830, 0.272822],
    [0.987053, 0.809468, 0.145202],
    [0.987622, 0.991102, 0.749504]
  ]
  
  const idx = t * (magmaColors.length - 1)
  const i = Math.floor(idx)
  const j = Math.min(i + 1, magmaColors.length - 1)
  const blend = idx - i
  
  const c1 = magmaColors[i]
  const c2 = magmaColors[j]
  
  const r = Math.round((c1[0] + (c2[0] - c1[0]) * blend) * 255)
  const g = Math.round((c1[1] + (c2[1] - c1[1]) * blend) * 255)
  const b = Math.round((c1[2] + (c2[2] - c1[2]) * blend) * 255)
  
  return `rgb(${r}, ${g}, ${b})`
}

// Dyad similarity function (simplified version of biotuner's metric)
function dyadSimilarity(ratio) {
  // Based on Tenney Height and harmonic entropy
  const log_ratio = Math.log2(ratio)
  const octave_reduced = log_ratio - Math.floor(log_ratio)
  
  // Simple consonances (octave-reduced)
  const consonances = [
    { ratio: 0, weight: 1.0 },      // 1/1 (unison)
    { ratio: Math.log2(3/2), weight: 0.95 },  // 3/2 (fifth)
    { ratio: Math.log2(4/3), weight: 0.9 },   // 4/3 (fourth)
    { ratio: Math.log2(5/4), weight: 0.85 },  // 5/4 (major third)
    { ratio: Math.log2(6/5), weight: 0.8 },   // 6/5 (minor third)
    { ratio: Math.log2(5/3), weight: 0.75 },  // 5/3
  ]
  
  let maxSimilarity = 0
  for (const cons of consonances) {
    const dist = Math.abs(octave_reduced - cons.ratio)
    const similarity = cons.weight * Math.exp(-dist * 8)
    maxSimilarity = Math.max(maxSimilarity, similarity)
  }
  
  return maxSimilarity
}

export default function ConsonanceMatrix({ tuning, maxDenominator = 100 }) {
  // Calculate consonance matrix
  const consonanceData = useMemo(() => {
    if (!tuning || tuning.length < 2) return null

    const matrix = []
    const size = Math.min(tuning.length, 12) // Limit to 12x12 for readability

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const ratio = tuning[i] / tuning[j]
        
        // Calculate consonance using dyad similarity
        const consonance = dyadSimilarity(ratio)
        
        matrix.push({
          x: i,
          y: j,
          ratio: ratio,
          ratioFraction: decimalToFraction(ratio, maxDenominator),
          consonance: consonance,
          freq1: decimalToFraction(tuning[i], maxDenominator),
          freq2: decimalToFraction(tuning[j], maxDenominator)
        })
      }
    }

    return { matrix, size }
  }, [tuning, maxDenominator])

  if (!consonanceData) return null

  const { matrix, size } = consonanceData
  const cellSize = 60

  return (
    <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-6">
      <h3 className="text-sm font-semibold text-biotuner-light/60 uppercase tracking-wider mb-4">
        Consonance Matrix
      </h3>
      
      <div className="overflow-x-auto">
        <svg width={size * cellSize + 80} height={size * cellSize + 80}>
          {/* Y-axis labels */}
          {Array.from({ length: size }).map((_, i) => (
            <text
              key={`y-${i}`}
              x={50}
              y={i * cellSize + cellSize / 2 + 65}
              textAnchor="end"
              dominantBaseline="middle"
              className="text-sm fill-biotuner-light/60"
            >
              {decimalToFraction(tuning[i], maxDenominator)}
            </text>
          ))}

          {/* X-axis labels */}
          {Array.from({ length: size }).map((_, i) => (
            <text
              key={`x-${i}`}
              x={i * cellSize + cellSize / 2 + 65}
              y={45}
              textAnchor="middle"
              className="text-sm fill-biotuner-light/60"
            >
              {decimalToFraction(tuning[i], maxDenominator)}
            </text>
          ))}

          {/* Matrix cells */}
          <g transform="translate(65, 65)">
            {matrix.map((cell, idx) => {
              // Use magma colormap based on consonance
              const color = magmaColor(cell.consonance)
              
              return (
                <g key={idx}>
                  <rect
                    x={cell.x * cellSize}
                    y={cell.y * cellSize}
                    width={cellSize - 2}
                    height={cellSize - 2}
                    fill={color}
                    className="hover:opacity-80 transition-opacity cursor-pointer"
                  >
                    <title>
                      {cell.freq1} / {cell.freq2} = {cell.ratioFraction}
                      {'\n'}Decimal: {cell.ratio.toFixed(3)}
                      {'\n'}Consonance: {(cell.consonance * 100).toFixed(1)}%
                    </title>
                  </rect>
                  
                  {/* Show fraction ratio in cells */}
                  <text
                    x={cell.x * cellSize + cellSize / 2}
                    y={cell.y * cellSize + cellSize / 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="text-[10px] fill-white font-semibold pointer-events-none"
                    style={{ textShadow: '0 0 3px black' }}
                  >
                    {cell.ratioFraction}
                  </text>
                </g>
              )
            })}
          </g>
        </svg>
      </div>

      <div className="mt-6 space-y-2">
        <span className="text-xs text-biotuner-light/60 font-semibold">Consonance:</span>
        <div className="h-8 rounded-lg border border-biotuner-dark-600" style={{
          background: 'linear-gradient(to right, rgb(0, 1, 4), rgb(87, 16, 110), rgb(188, 55, 84), rgb(249, 142, 9), rgb(252, 253, 191))'
        }}></div>
        <div className="flex justify-between text-xs text-biotuner-light/60">
          <span>Low</span>
          <span>High</span>
        </div>
      </div>
    </div>
  )
}
