"""Carbon footprint tracking for agent evaluations."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timezone


@dataclass
class CarbonReport:
    """Carbon emissions report.
    
    Attributes:
        emissions_kg: Total emissions in kg CO2eq
        energy_kwh: Total energy consumed in kWh
        duration_seconds: Measurement duration
        emissions_rate: Emissions rate (kg CO2/hour)
        grid_intensity: Carbon intensity of electricity (kg CO2/kWh)
        timestamp: Report timestamp
    """
    emissions_kg: float
    energy_kwh: float
    duration_seconds: float
    emissions_rate: float
    grid_intensity: float
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class CarbonTracker:
    """Track carbon footprint of agent evaluations.
    
    Integrates with CodeCarbon when available, otherwise uses
    simplified estimation based on resource usage.
    """
    
    def __init__(
        self,
        grid_intensity: float = 0.475,  # kg CO2/kWh (global avg)
        use_codecarbon: bool = True
    ):
        """Initialize carbon tracker.
        
        Args:
            grid_intensity: Carbon intensity of electricity grid
            use_codecarbon: Whether to use CodeCarbon if available
        """
        self.grid_intensity = grid_intensity
        self.use_codecarbon = use_codecarbon
        
        # Try to import CodeCarbon
        self.codecarbon_tracker = None
        if use_codecarbon:
            try:
                from codecarbon import EmissionsTracker
                self.codecarbon_tracker = EmissionsTracker()
            except ImportError:
                pass
        
        self.start_time: Optional[float] = None
        self.total_energy_kwh: float = 0.0
    
    def start(self):
        """Start tracking carbon emissions."""
        import time
        self.start_time = time.time()
        self.total_energy_kwh = 0.0
        
        if self.codecarbon_tracker:
            self.codecarbon_tracker.start()
    
    def update(self, energy_kwh: float):
        """Update energy consumption.
        
        Args:
            energy_kwh: Energy consumed since last update
        """
        self.total_energy_kwh += energy_kwh
    
    def stop(self) -> CarbonReport:
        """Stop tracking and generate report.
        
        Returns:
            CarbonReport with emissions data
        """
        import time
        
        duration = 0.0
        if self.start_time:
            duration = time.time() - self.start_time
        
        # Use CodeCarbon data if available
        if self.codecarbon_tracker:
            emissions = self.codecarbon_tracker.stop()
            # CodeCarbon returns emissions in kg
            emissions_kg = emissions if isinstance(emissions, float) else 0.0
            energy_kwh = emissions_kg / self.grid_intensity if self.grid_intensity > 0 else 0.0
        else:
            # Use accumulated energy
            energy_kwh = self.total_energy_kwh
            emissions_kg = energy_kwh * self.grid_intensity
        
        # Calculate emissions rate
        emissions_rate = 0.0
        if duration > 0:
            emissions_rate = (emissions_kg / duration) * 3600  # kg CO2/hour
        
        return CarbonReport(
            emissions_kg=emissions_kg,
            energy_kwh=energy_kwh,
            duration_seconds=duration,
            emissions_rate=emissions_rate,
            grid_intensity=self.grid_intensity
        )
    
    def get_report(self) -> Dict[str, Any]:
        """Get detailed carbon report.
        
        Returns:
            Dictionary with carbon footprint details
        """
        report = self.stop()
        
        return {
            "emissions": {
                "kg_co2": report.emissions_kg,
                "g_co2": report.emissions_kg * 1000,
                "rate_kg_per_hour": report.emissions_rate
            },
            "energy": {
                "kwh": report.energy_kwh,
                "wh": report.energy_kwh * 1000
            },
            "intensity": {
                "kg_co2_per_kwh": report.grid_intensity
            },
            "duration": {
                "seconds": report.duration_seconds,
                "minutes": report.duration_seconds / 60
            },
            "equivalents": {
                "km_driven": report.emissions_kg * 4.6,  # Avg car: 0.22 kg CO2/km
                "trees_needed": report.emissions_kg / 21,  # Avg tree absorbs 21 kg CO2/year
                "smartphones_charged": report.energy_kwh * 121  # Avg smartphone: 0.00826 kWh
            },
            "source": "codecarbon" if self.codecarbon_tracker else "estimated"
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
