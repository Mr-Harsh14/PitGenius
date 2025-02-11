# Visualization Module API Reference

The visualization module provides tools for creating insightful visualizations of race data, pit stop strategies, and model performance metrics.

## PitStopVisualizer

`src.visualization.pit_stop_viz.PitStopVisualizer`

Class for creating visualizations related to pit stop strategies and race analysis.

### Methods

#### `__init__(style: str = 'dark')`
Initialize the visualizer.

**Parameters:**
- `style`: Plot style ('dark' or 'light')

#### `plot_pit_windows(race_data: pd.DataFrame, predictions: pd.DataFrame) -> plt.Figure`
Plot predicted pit windows overlaid on race data.

**Parameters:**
- `race_data`: DataFrame with race telemetry
- `predictions`: DataFrame with pit window predictions

**Returns:**
- Matplotlib figure object

#### `plot_tire_degradation(race_data: pd.DataFrame) -> plt.Figure`
Visualize tire degradation over race distance.

**Parameters:**
- `race_data`: DataFrame with tire data

**Returns:**
- Matplotlib figure object

#### `plot_strategy_comparison(race_data: pd.DataFrame) -> plt.Figure`
Compare different pit stop strategies.

**Parameters:**
- `race_data`: DataFrame with race strategies

**Returns:**
- Matplotlib figure object

#### `plot_position_changes(race_data: pd.DataFrame) -> plt.Figure`
Visualize position changes throughout the race.

**Parameters:**
- `race_data`: DataFrame with position data

**Returns:**
- Matplotlib figure object

#### `create_race_summary(race_data: pd.DataFrame) -> plt.Figure`
Create comprehensive race summary visualization.

**Parameters:**
- `race_data`: DataFrame with race data

**Returns:**
- Matplotlib figure with subplots

## Interactive Visualizations

### Methods

#### `create_strategy_dashboard(race_data: pd.DataFrame) -> None`
Create interactive Streamlit dashboard for strategy analysis.

**Parameters:**
- `race_data`: DataFrame with race data

#### `plot_interactive_timeline(race_data: pd.DataFrame) -> None`
Create interactive timeline of race events.

**Parameters:**
- `race_data`: DataFrame with race events

## Plot Customization

### Color Schemes

```python
DARK_THEME = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'grid': '#333333',
    'highlight': '#FF1E1E'
}

LIGHT_THEME = {
    'background': '#FFFFFF',
    'text': '#000000',
    'grid': '#CCCCCC',
    'highlight': '#FF0000'
}
```

### Plot Types

1. **Pit Window Visualization**
   - Optimal pit windows highlighted
   - Tire compound indicators
   - Gap to competitors

2. **Tire Performance**
   - Degradation curves
   - Compound comparison
   - Temperature impact

3. **Race Progress**
   - Position changes
   - Lap time evolution
   - Strategy impact

## Usage Examples

### Basic Pit Stop Visualization

```python
from src.visualization.pit_stop_viz import PitStopVisualizer

visualizer = PitStopVisualizer(style='dark')
fig = visualizer.plot_pit_windows(race_data, predictions)
fig.savefig('pit_windows.png')
```

### Interactive Dashboard

```python
from src.visualization.pit_stop_viz import PitStopVisualizer

visualizer = PitStopVisualizer()
visualizer.create_strategy_dashboard(race_data)
```

### Custom Plot Styling

```python
from src.visualization.pit_stop_viz import PitStopVisualizer

visualizer = PitStopVisualizer()
fig = visualizer.plot_tire_degradation(race_data)
visualizer.customize_plot(fig, 
    title="Tire Degradation Analysis",
    xlabel="Lap Number",
    ylabel="Degradation (%)"
)
```

## Plot Configurations

### Default Plot Settings

```python
PLOT_CONFIG = {
    'figure.figsize': (12, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'legend.fontsize': 10
}
```

### Animation Settings

```python
ANIMATION_CONFIG = {
    'interval': 100,  # milliseconds
    'blit': True,
    'repeat': False
}
```

## Output Formats

The visualization module supports multiple output formats:

1. **Static Images**
   - PNG (high resolution)
   - PDF (vector graphics)
   - SVG (web-friendly)

2. **Interactive Plots**
   - HTML (Plotly)
   - Streamlit components
   - Jupyter widgets

3. **Animations**
   - GIF
   - MP4
   - Interactive HTML 