"""
Color Palette Service
Converts tuning to color palettes using biotuner's biocolors module
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Any, Optional
import sys
from pathlib import Path
from struct import pack

# Add parent biotuner package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from biotuner.biocolors import audible2visible, scale2freqs, wavelength_to_rgb
from biotuner.metrics import tuning_cons_matrix, dyad_similarity


class ColorService:
    """Service for color palette generation from tuning"""
    
    def tuning_to_colors(
        self,
        tuning: List[float],
        fundamental: float = 440.0
    ) -> Dict[str, Any]:
        """
        Convert tuning to color palette
        
        Parameters
        ----------
        tuning : list
            Tuning ratios
        fundamental : float
            Fundamental frequency
            
        Returns
        -------
        dict : Color palette with hex values and names
        """
        try:
            # Convert tuning to frequencies (in Hz, not THz)
            scale_freqs = scale2freqs(tuning, fundamental, THz=False)
            
            # Compute averaged consonance of each step against all others
            # This matches the biotuner viz_scale_colors implementation
            # tuning_cons_matrix returns: (metric_values_per_step, metric_avg, full_matrix)
            scale_cons, _, _ = tuning_cons_matrix(tuning, dyad_similarity, ratio_type='all')
            scale_cons = np.array(scale_cons)
            
            # Debug: check consonance range
            print(f"Consonance values: min={scale_cons.min()}, max={scale_cons.max()}, values={scale_cons}")
            
            # Normalize consonance to 0-1 range if needed
            # dyad_similarity can return values > 1
            if scale_cons.max() > 1.0:
                scale_cons = scale_cons / scale_cons.max()
            
            # Clamp to 0-1 range
            scale_cons = np.clip(scale_cons, 0.0, 1.0)
            
            colors = {}
            
            for i, (freq, cons) in enumerate(zip(scale_freqs, scale_cons)):
                # Convert frequency to wavelength
                try:
                    wavelength_result = audible2visible(float(freq))
                    # audible2visible returns (THz, Hz, nm, n_octave)
                    # We need the nm value (index 2)
                    if isinstance(wavelength_result, (tuple, list)) and len(wavelength_result) >= 3:
                        wavelength = float(wavelength_result[2])  # nm value
                    else:
                        wavelength = float(wavelength_result)
                except Exception as e:
                    print(f"Warning: Could not convert freq {freq} to wavelength: {e}")
                    wavelength = 550.0  # Default to green (550 nm)
                
                # Convert wavelength to RGB
                try:
                    rgb_result = wavelength_to_rgb(wavelength)
                    # Ensure RGB is a tuple of integers
                    if isinstance(rgb_result, tuple) and len(rgb_result) >= 3:
                        rgb = tuple(int(float(x)) for x in rgb_result[:3])
                    else:
                        rgb = (128, 128, 128)  # Default gray
                except Exception as e:
                    print(f"Warning: Could not convert wavelength {wavelength} to RGB: {e}")
                    rgb = (128, 128, 128)  # Default gray
                
                # Ensure consonance is a scalar float
                try:
                    cons_value = float(cons)
                except Exception as e:
                    print(f"Warning: Could not convert consonance {cons} to float: {e}")
                    cons_value = 0.5
                
                # Adjust saturation based on consonance, fix brightness like v1
                hsv = self._rgb_to_hsv(rgb)
                # Match v1 implementation: consonance controls saturation, fixed brightness at 200/255
                hsv = (hsv[0], cons_value, 200/255)  # Saturation=consonance, Value=200/255 (fixed)
                rgb = self._hsv_to_rgb(hsv)
                
                # Convert to hex
                hex_color = self._rgb_to_hex(rgb)
                
                # Generate name
                color_name = f"Note_{i+1}_{freq:.2f}Hz"
                
                colors[color_name] = {
                    'hex': hex_color,
                    'rgb': rgb,
                    'frequency': float(freq),
                    'consonance': cons_value,
                    'wavelength': float(wavelength)
                }
            
            return {
                'palette': colors,
                'fundamental': fundamental,
                'n_colors': len(colors)
            }
        
        except Exception as e:
            print(f"Error generating colors: {str(e)}")
            raise
    
    def generate_palette_from_peaks(
        self,
        peaks: List[float],
        powers: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Generate color palette directly from peak frequencies
        Uses octave multiplication to map to visible spectrum (same as biotuner)
        
        Parameters
        ----------
        peaks : List[float]
            Peak frequencies in Hz
        powers : List[float], optional
            Peak powers/amplitudes for saturation control
            
        Returns
        -------
        dict : Color palette with hex values and names
        """
        try:
            peaks = np.array(peaks)
            
            # Check if peaks array is empty
            if len(peaks) == 0:
                raise ValueError("No peaks available to generate palette")
            
            # Normalize powers for saturation if provided
            if powers is not None and len(powers) > 0:
                powers = np.array(powers)
                # Ensure powers array matches peaks length
                if len(powers) != len(peaks):
                    print(f"Warning: Powers length ({len(powers)}) doesn't match peaks length ({len(peaks)}), using default saturation")
                    saturations = np.ones_like(peaks) * 0.7
                else:
                    # Normalize to 0.4-1.0 range for better visibility
                    min_pow = powers.min()
                    max_pow = powers.max()
                    if max_pow > min_pow:
                        saturations = 0.4 + 0.6 * (powers - min_pow) / (max_pow - min_pow)
                    else:
                        saturations = np.ones_like(powers) * 0.7
            else:
                # Use default saturation
                saturations = np.ones_like(peaks) * 0.7
            
            colors = {}
            
            for i, (peak, saturation) in enumerate(zip(peaks, saturations)):
                # Use audible2visible to map frequency to visible spectrum via octave multiplication
                # This is the biotuner way - multiplies by 2^n until it reaches visible range
                try:
                    wavelength_result = audible2visible(float(peak))
                    # audible2visible returns (THz, Hz, nm, n_octave)
                    if isinstance(wavelength_result, (tuple, list)) and len(wavelength_result) >= 3:
                        wavelength = float(wavelength_result[2])  # nm value
                    else:
                        wavelength = float(wavelength_result)
                except Exception as e:
                    print(f"Warning: Could not convert freq {peak} to wavelength: {e}")
                    wavelength = 550.0
                
                # Convert wavelength to RGB
                try:
                    rgb_result = wavelength_to_rgb(wavelength)
                    if isinstance(rgb_result, tuple) and len(rgb_result) >= 3:
                        rgb = tuple(int(float(x)) for x in rgb_result[:3])
                    else:
                        rgb = (128, 128, 128)
                except Exception as e:
                    print(f"Warning: Could not convert wavelength {wavelength} to RGB: {e}")
                    rgb = (128, 128, 128)
                
                # Adjust saturation based on power, fix brightness like v1
                hsv = self._rgb_to_hsv(rgb)
                hsv = (hsv[0], float(saturation), 200/255)  # Saturation from power, Value=200/255
                rgb = self._hsv_to_rgb(hsv)
                
                # Convert to hex
                hex_color = self._rgb_to_hex(rgb)
                
                # Generate name
                color_name = f"Peak_{i+1}_{peak:.2f}Hz"
                
                colors[color_name] = {
                    'hex': hex_color,
                    'rgb': rgb,
                    'frequency': float(peak),
                    'saturation': float(saturation),
                    'wavelength': float(wavelength)
                }
            
            return {
                'palette': colors,
                'n_colors': len(colors)
            }
        
        except Exception as e:
            print(f"Error generating palette from peaks: {e}")
            raise
    
    def export_palette(
        self,
        colors: Dict[str, str],
        format: str,
        filename: str = "palette"
    ) -> bytes:
        """
        Export color palette to various formats
        
        Parameters
        ----------
        colors : dict
            Dictionary of color names to hex values
        format : str
            Export format (ase, json, svg, css, gpl)
        filename : str
            Output filename
            
        Returns
        -------
        bytes : File data
        """
        if format == "json":
            return self._export_json(colors)
        elif format == "svg":
            return self._export_svg(colors)
        elif format == "css":
            return self._export_css(colors)
        elif format == "gpl":
            return self._export_gpl(colors)
        elif format == "ase":
            return self._export_ase(colors)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, colors: Dict[str, str]) -> bytes:
        """Export as JSON"""
        data = {
            "colors": [
                {"name": name, "hex": hex_val}
                for name, hex_val in colors.items()
            ]
        }
        return json.dumps(data, indent=2).encode('utf-8')
    
    def _export_svg(self, colors: Dict[str, str]) -> bytes:
        """Export as SVG"""
        width = 500
        rect_width = width // max(1, len(colors))
        
        rectangles = []
        for i, (name, hex_color) in enumerate(colors.items()):
            x = i * rect_width
            rect = f'<rect x="{x}" y="0" width="{rect_width}" height="100" fill="{hex_color}"/>'
            rectangles.append(rect)
        
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="100" xmlns="http://www.w3.org/2000/svg">
    {chr(10).join(rectangles)}
</svg>'''
        
        return svg.encode('utf-8')
    
    def _export_css(self, colors: Dict[str, str]) -> bytes:
        """Export as CSS"""
        css = "/* Generated Color Palette */\n:root {\n"
        for name, hex_color in colors.items():
            var_name = name.lower().replace(' ', '-')
            css += f"    --{var_name}: {hex_color};\n"
        css += "}"
        
        return css.encode('utf-8')
    
    def _export_gpl(self, colors: Dict[str, str]) -> bytes:
        """Export as GIMP Palette"""
        gpl = "GIMP Palette\nName: CustomPalette\n#\n"
        
        for name, hex_color in colors.items():
            rgb = self._hex_to_rgb(hex_color)
            gpl += f"{rgb[0]:3d} {rgb[1]:3d} {rgb[2]:3d}  {name}\n"
        
        return gpl.encode('utf-8')
    
    def _export_ase(self, colors: Dict[str, str]) -> bytes:
        """Export as Adobe Swatch Exchange"""
        # Simplified ASE format
        data = b'ASEF'  # Signature
        data += pack('>HH', 1, 0)  # Version
        data += pack('>I', len(colors))  # Number of blocks
        
        for name, hex_color in colors.items():
            rgb = self._hex_to_rgb(hex_color)
            # Color entry block
            data += pack('>H', 1)  # Block type (color entry)
            name_bytes = name.encode('utf-16be')
            data += pack('>I', len(name_bytes) + 18)  # Block length
            data += pack('>H', len(name) + 1)  # Name length
            data += name_bytes + b'\x00\x00'
            data += b'RGB '  # Color model
            data += pack('>fff', rgb[0]/255, rgb[1]/255, rgb[2]/255)
            data += pack('>H', 2)  # Color type (global)
        
        return data
    
    # Color conversion utilities
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex to RGB"""
        try:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                raise ValueError(f"Invalid hex color: {hex_color}")
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except Exception as e:
            print(f"Error converting hex to RGB: {hex_color}, error: {e}")
            return (128, 128, 128)  # Default gray
    
    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV"""
        r, g, b = [x / 255.0 for x in rgb]
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c
        
        if max_c == min_c:
            h = 0
        elif max_c == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_c == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        s = 0 if max_c == 0 else (diff / max_c)
        v = max_c
        
        return (h, s, v)
    
    def _hsv_to_rgb(self, hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert HSV to RGB"""
        h, s, v = hsv
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255)
        )
