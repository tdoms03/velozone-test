import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Optionally use seaborn
from fpdf import FPDF
from datetime import datetime
# import urllib3
from data_analysis import remove_initial_lap, preprocess_lap_times, diesel_engine_df

# ------------------------------------------------------------
# 1. Load & Preprocess Data
# ------------------------------------------------------------
def load_and_preprocess_data(csv_file_path):
    # Read CSV
    df = pd.read_csv(csv_file_path)
    
    # Convert timestamps to datetime
    df['utcTimestamp'] = pd.to_numeric(df['utcTimestamp'], errors='coerce')
    
    # Drop duplicates if desired
    df.drop_duplicates(inplace=True)
    
    # Drop rows with missing critical values (e.g., transponder_id or loop)
    df.dropna(subset=['transponder_id', 'loop', 'utcTimestamp'], inplace=True)

    # df = remove_initial_lap(df)
    df = preprocess_lap_times(df)
    
    # Sort by transponder and timestamp
    # df.sort_values(by=['transponder_id', 'timestamp'], inplace=True)
    
    return df

# ------------------------------------------------------------
# 2. Compute Key Metrics for each rider
# ------------------------------------------------------------
def compute_metrics(df, track_length=250, loop_filter='L01'):
    """
    Calculates, per rider (transponder_id):
      - total laps
      - total distance (meters)
      - fastest lap time (s)
      - avg lap time (s)

    Also computes group-level averages.
    
    If the dataset includes lapTimes for multiple loops 
    but only L01 is the "start/finish" loop, we can optionally filter.
    """
    # If needed, filter only the loop that signifies a completed lap.
    if loop_filter and 'loop' in df.columns:
        df_filtered = df[df['loop'] == loop_filter].copy()
    else:
        df_filtered = df.copy()
    
    summary_list = []
    # Group by transponder_id and run over all of the riders
    for rider_id, rider_df in df_filtered.groupby('transponder_id'):
        total_laps = len(rider_df)
        total_distance = total_laps * track_length
        
        # Provided lapTime is already the time for a full lap
        fastest_lap = rider_df['lapTime'].min()
        avg_lap_time = rider_df['lapTime'].mean()
        
        summary_list.append({
            'transponder_id': rider_id,
            'total_laps': total_laps,
            'total_distance_m': total_distance,
            'fastest_lap_s': fastest_lap,
            'avg_lap_time_s': avg_lap_time
        })
    
    summary_df = pd.DataFrame(summary_list)
    
    # Compute group-level stats
    group_stats = {
        'group_avg_distance_m': summary_df['total_distance_m'].mean() if not summary_df.empty else np.nan,
        'group_avg_fastest_lap_s': summary_df['fastest_lap_s'].mean() if not summary_df.empty else np.nan,
        'group_avg_lap_time_s': summary_df['avg_lap_time_s'].mean() if not summary_df.empty else np.nan
    }
    
    return summary_df, group_stats

# ------------------------------------------------------------
# 3. Statistics from students
# ------------------------------------------------------------
def general_stats(df, track_length=250, loop_filter='L01'):
    # If needed, filter only the loop that signifies a completed lap.
    if loop_filter and 'loop' in df.columns:
        df_filtered = df[df['loop'] == loop_filter].copy()
    else:
        df_filtered = df.copy()

    # Badman
    badman = df_filtered.loc[df_filtered['lapTime'].idxmax(),['transponder_id','lapTime']].to_frame().T
    badman.columns = ['transponder_id', 'worst_lap_time']

    # Diesel Engine
    diesel_engine = diesel_engine_df(df_filtered)

    return badman, diesel_engine

# ------------------------------------------------------------
# 4. Generate Time-Series Plot
# ------------------------------------------------------------
def generate_lap_time_plot(rider_id, rider_df, group_stats, output_folder='plots'):
    """
    Plots the rider's lapTime vs. timestamp (if available). 
    Overlays a horizontal line for the group average lap time.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if rider_df.empty:
        return None
    
    # For plotting, we assume the data has a 'timestamp' column.
    # If your CSV doesn't, you can plot lap index on the x-axis instead.
    
    # Sort by timestamp for proper time-series
    if 'utcTimestamp' in rider_df.columns:
        rider_df = rider_df.sort_values('utcTimestamp')
        x_values = rider_df['utcTimestamp'] - rider_df['utcTimestamp'].min()
        x_label = 'Time [s]'
    else:
        # fallback: use enumeration as x-axis
        rider_df = rider_df.reset_index(drop=True)
        x_values = rider_df.index
        x_label = 'Lap Index'
    
    y_laptimes = rider_df['lapTime']
    y_avg_laptime = rider_df['lapTime'].mean()
    
    plt.figure(figsize=(8,4))
    plt.scatter(x_values, y_laptimes)
    
    # Group average lap time
    group_avg_lap_time = group_stats['group_avg_lap_time_s']
    if not np.isnan(group_avg_lap_time):
        plt.axhline(y=group_avg_lap_time, color='r', linestyle='--', label='Group Avg Lap Time')
    plt.axhline(y=y_avg_laptime, color='g', linestyle='--', label='Rider Avg Lap Time')
    
    plt.title(f'Lap Times for Rider {rider_id}')
    plt.xlabel(x_label)
    plt.ylabel('Lap Time [s]')
    plt.legend()
    plt.tight_layout()
    plt.ticklabel_format(axis='x', style='plain')
    
    plot_filename = os.path.join(output_folder, f'{rider_id}_lap_time_plot.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    return plot_filename

# I like this PLOT --> do not need to change in my opinion
def generate_fastest_lap_comparison_plot(rider_id, summary_df, output_folder='plots'):
    """
    Creates a bar chart showing each rider's fastest lap time, 
    highlighting the current rider in green, others in gray,
    with numeric labels above each bar.
    Returns path to the saved plot image.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Sort riders by fastest lap time ascending for a neat bar chart
    sorted_df = summary_df.sort_values('fastest_lap_s').reset_index(drop=True)
    
    # Plot config
    plt.figure(figsize=(8, 5))
    
    # Determine bar colors
    bar_colors = []
    for tid in sorted_df['transponder_id']:
        if tid == rider_id:
            bar_colors.append('green')
        else:
            bar_colors.append('gray')
    
    x_positions = np.arange(len(sorted_df))
    y_values = sorted_df['fastest_lap_s']
    
    bars = plt.bar(x_positions, y_values, color=bar_colors)
    
    # Add numeric labels above bar of current rider
    rider_bar = bars[sorted_df[sorted_df['transponder_id'] == rider_id].index[0]]
    plt.text(
        rider_bar.get_x() + rider_bar.get_width()/2,  # center of the bar
        rider_bar.get_height() + 0.2,                 # slight offset above top
        f"{rider_bar.get_height():.2f}",              # formatted fastest lap time
        ha='center',
        va='bottom',
        fontsize=8
    )

    # Add numeric labels above bars
    # for idx, bar in enumerate(bars):
    #     height = bar.get_height()
    #     plt.text(
    #         bar.get_x() + bar.get_width()/2,  # center of the bar
    #         height + 0.2,                    # slight offset above top
    #         f"{height:.2f}",                 # formatted fastest lap time
    #         ha='center',
    #         va='bottom',
    #         fontsize=8
    #     )
    
    # X-axis with rider IDs (or abbreviate if large)
    rider_labels = sorted_df['transponder_id'].astype(str)
    xtick_colors = ['green' if tid == rider_id else 'black' for tid in sorted_df['transponder_id']]
    plt.xticks(x_positions, rider_labels, rotation=45, ha='right')
    plt.tick_params(axis='x', which='major', labelsize=6)

    # Set rider tag in green
    plt.gca().get_xticklabels()[rider_labels[rider_labels == rider_id].index[0]].set_color('green')
    plt.xlabel('Rider ID')
    plt.ylabel('Fastest Lap (s)')
    plt.title('Fastest Lap Comparison')
    plt.tight_layout()
    
    plot_filename = os.path.join(output_folder, f'{rider_id}_fastest_lap_comparison.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    return plot_filename

def generate_speed_over_time_plot(rider_id, df, track_length=250, output_folder='plots'):
    """
    Plots the speed [m/s] of each rider over session time in light gray,
    with the current rider's speed in green.
    Applies a simple smoothing for aesthetics.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if df.empty:
        return None
    
    plt.figure(figsize=(8, 5))
    
    # Each rider: compute speed for each row, sort by time,
    # optionally apply smoothing, and plot in gray except the current.
    for tid, rider_df in df.groupby('transponder_id'):
        # Sort by time
        r = rider_df.sort_values('utcTimestamp')
        # x-values: offset to session start
        t = r['utcTimestamp'] - r['utcTimestamp'].min()
        # speed = track_length / lapTime
        # handle any zero or negative lapTime if it occurs
        lap_times = r['lapTime'].replace(0, np.nan) 
        speed_m_s = track_length / lap_times * 3.6  # [kph]
        
        speed_m_s = speed_m_s.fillna(0)  # replace NaNs with 0 for plotting
        
        # SMOOTH the speed array for aesthetics
        # Option A: rolling mean
        # speed_smoothed = speed_m_s.rolling(window=3, min_periods=1, center=True).mean()
        # Option B: Gaussian filter
        speed_smoothed = gaussian_filter1d(speed_m_s, sigma=1)
        
        if tid == rider_id:
            # current rider in green
            plt.plot(t, speed_m_s, color='green', linewidth=2.0, label=f'Rider {rider_id}')
        else:
            # all others in gray with alpha
            plt.plot(t, speed_m_s, color='gray', alpha=0.3, linewidth=1.0)
    
    plt.title(f'Speed Over Time (kph)')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [kph]')
    plt.legend()
    plt.tight_layout()
    
    plot_filename = os.path.join(output_folder, f'{rider_id}_speed_time_plot.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    return plot_filename

# ------------------------------------------------------------
# 5. Generate PDF Report
# ------------------------------------------------------------
class PDFReport(FPDF):
    def header(self):
        # Optional: Add a header with a logo or text
        self.set_font('Arial', 'B', 12)
        # Calculate positions for logos
        logos = [
            'media/logo-sport-vlaanderen.png',
            'media/logo-cycling-vlaanderen-horizontal.jpg',
            'media/logo-belgian-cycling.png'
        ]
        # Compute logo width for height 10
        logo_width = 10 * 1.5  # 1.5 aspect ratio
        spacing = (self.w - len(logos) * logo_width) / (len(logos) + 1)
        
        for i, logo in enumerate(logos):
            x_position = spacing + i * (logo_width + spacing)
            self.image(logo, x=x_position, y=10, h=10)

        self.set_y(10 + logo_width)
        self.ln()

    def footer(self):
        """Adds a footer with page number and a professional touch."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_table(self, data, col_widths, headers):
        """Adds a stylized table with alternating row colors and blue theme."""
        self.set_font("Arial", "B", 12)
        self.set_fill_color(0, 102, 204)  # Blue header
        self.set_text_color(255)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, header, border=1, align="C", fill=True)
        self.ln()

        self.set_font("Arial", "", 10)
        self.set_text_color(0)
        
        for i, row in enumerate(data):
            fill = i % 2 == 0  # Alternate row colors
            if fill:
                self.set_fill_color(173, 216, 230)  # Light blue background
            for j, cell in enumerate(row):
                formatted_cell = f'{cell:.2f}' if isinstance(cell, float) else str(cell)
                self.cell(col_widths[j], 8, formatted_cell, border=1, align="C", fill=fill)
            self.ln()

def create_rider_pdf_report(
    rider_id, summary_row, group_stats, 
    lap_time_plot_path, fastest_lap_plot_path, speed_time_plot_path,
    output_dir='output_reports', event_name=None, event_date=datetime.now().strftime('%Y-%m-%d')
):
    image_width = 160

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # -- Report Title --
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 16, 'Sport.Vlaanderen - Wielercentrum Eddy Merckx', 0, 1, 'C')
    pdf.ln(5)

    # -- Add Group/Event Logo --
    logo_width = 60
    pdf.set_x((pdf.w - logo_width) / 2)
    pdf.image('media/logo-idlab.jpg', w=logo_width)
    pdf.ln(5)  

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f'{event_name} - {event_date}', ln=True, align='C')
    pdf.ln(5)
    pdf.cell(0, 10, f'Summary for {rider_id}', ln=True, align='C')
    pdf.ln(5)

 
    
    # -- Key Stats --
    pdf.set_font('Arial', '', 12)
    total_distance = summary_row['total_distance_m']
    fastest_lap = summary_row['fastest_lap_s']
    avg_lap_time = summary_row['avg_lap_time_s']
    
    group_avg_distance = group_stats['group_avg_distance_m']
    group_avg_fastest_lap = group_stats['group_avg_fastest_lap_s']
    group_avg_lap_time = group_stats['group_avg_lap_time_s']
    
    pdf.cell(0, 10, f"Total Distance: {total_distance:.2f} m (Group Avg: {group_avg_distance:.2f} m)", ln=True, align='C')
    pdf.cell(0, 10, f"Fastest Lap: {fastest_lap:.2f} s (Group Avg Fastest Lap: {group_avg_fastest_lap:.2f} s)", ln=True, align='C')
    pdf.cell(0, 10, f"Average Lap Time: {avg_lap_time:.2f} s (Group Avg: {group_avg_lap_time:.2f} s)", ln=True, align='C')
    
    # -- Summary Lines --
    pdf.ln(5)
    pdf.set_font('Arial', 'I', 12)
    distance_diff = total_distance - group_avg_distance
    fastest_diff = fastest_lap - group_avg_fastest_lap
    average_diff = avg_lap_time - group_avg_lap_time
    
    summary_text = [
        f"Distance vs. Group: {'+' if distance_diff >= 0 else '-'}{abs(distance_diff):.2f} m",
        f"Fastest Lap vs. Group: {'+' if fastest_diff >= 0 else '-'}{abs(fastest_diff):.2f} s",
        f"Avg Lap Time vs. Group: {'+' if average_diff >= 0 else '-'}{abs(average_diff):.2f} s"
    ]
    
    for line in summary_text:
        pdf.cell(0, 10, line, ln=True, align='C')

    # Define plot parameters
    x = (pdf.w - image_width) / 2 # To center the plot
    
    # -- Lap-Time Plot --
    if lap_time_plot_path and os.path.exists(lap_time_plot_path):
        pdf.cell(0, 10, 'Lap Times', ln=True, align='C')
        pdf.ln(5)
        pdf.image(lap_time_plot_path, x=x, w=image_width)
    else:
        pdf.cell(0, 10, "No lap-time plot available.", ln=True)
    
    # -- Fastest Lap Comparison Plot --
    if fastest_lap_plot_path and os.path.exists(fastest_lap_plot_path):
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Fastest Laps Per Rider', ln=True, align='C')
        pdf.ln(5)
        pdf.image(fastest_lap_plot_path, x=x, w=image_width)
    else:
        pdf.cell(0, 10, "No fastest-lap comparison chart available.", ln=True)
    
    # -- Speed Over Time Plot (NEW) --
    if speed_time_plot_path and os.path.exists(speed_time_plot_path):
        # pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Speed During Session', ln=True, align='C')
        pdf.ln(5)
        pdf.image(speed_time_plot_path, x=x, w=image_width)
    else:
        pdf.cell(0, 10, "No speed-time plot available.", ln=True)
    
    # Save final PDF
    output_path = os.path.join(output_dir, f"rider_report_{rider_id}.pdf")
    pdf.output(output_path)

def create_general_report(group_name,summary_df, group_stats,badman, diesel_engine, 
    output_dir='output_reports', event_name=None, event_date=datetime.now().strftime('%Y-%m-%d')):
    """
    Creates a general report for the whole session of the group

    Parameters:
    -----------
    group_name (str): Name of the group
    summary_df (DataFrame): Summary statistics for all riders
    group_stats (dict): Group-level statistics
    badman (DataFrame): Dataframe containing the worst rider in the group
    diesel_engine (DataFrame): Dataframe containing the diesel engine data for the group
    output_dir (str): Directory where the report should be saved
    event_name (str): Name of the event
    event_date (str): Date of the event

    Returns:
    --------
    Creates the report in the correct directory
    """
    image_width = 160

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 16, 'Sport.Vlaanderen - Wielercentrum Eddy Merckx', 0, 1, 'C')
    pdf.ln(5)
    
    pdf.image('media/logo-idlab.jpg', x=(pdf.w - 60) / 2, w=60)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f'{event_name} - {event_date}', ln=True, align='C')
    pdf.ln(5)
    pdf.cell(0, 10, f'Summary for {group_name}', ln=True, align='C')
    pdf.ln(10)

    pdf.set_font('Arial', '', 12)
    pdf.set_fill_color(200, 220, 255)  # Light blue for key statistics
    pdf.cell(0, 8, f'Worst lap: {badman.iloc[0,0]} - {badman.iloc[0,1]:.2f}s', ln=True, align='C', fill=True)
    pdf.ln(4)
    pdf.cell(0, 8, f'Most Consistent Rider: {diesel_engine.iloc[0,0]} - {diesel_engine.iloc[0,2]:.2f}s', ln=True, align='C', fill=True)
    pdf.ln(4)
    best_rider = summary_df.nsmallest(1, 'avg_lap_time_s')
    pdf.cell(0, 8, f'Best Rider: {best_rider.iloc[0,0]} - {best_rider.iloc[0,4]:.2f}s - {best_rider.iloc[0,1]} laps', ln=True, align='C', fill=True)
    pdf.ln(10)
    
    image_x = (pdf.w - image_width) / 2 # To center the plot
    image_y = pdf.get_y()

    # Top 3  Riders
    top_riders = summary_df.nlargest(3,'total_laps').reset_index(drop = True)
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(0, 102, 204)  # Blue for title
    pdf.set_text_color(0)
    pdf.cell(0, 10, "Podium - Most Laps", ln=True, align='C', fill=True)
    pdf.ln(10)
    # Image of the podium
    pdf.image('media\podium.png', x=image_x, w=image_width)
    pdf.ln(5)

    # Define text positions relative to the podium image
    first_x = image_x + image_width / 2 - 23
    first_y = image_y + 20   # Raise text above the first-place podium

    second_x = image_x + image_width * 0.04
    second_y = image_y + 45  # Adjust to align with second place

    third_x = image_x + image_width * 0.70
    third_y = image_y + 65  # Adjust to align with third place

    # Draw rider names above the correct podium positions
    pdf.text(first_x, first_y, f"{top_riders.iloc[0, 0]} ({top_riders.iloc[0, 1]} laps)")
    pdf.text(second_x, second_y, f"{top_riders.iloc[1, 0]} ({top_riders.iloc[1, 1]} laps)")
    pdf.text(third_x, third_y, f"{top_riders.iloc[2, 0]} ({top_riders.iloc[2, 1]} laps)")

    # Create the table
    sorted_by_nr_laps = summary_df.sort_values(by='total_laps', ascending=False).reset_index(drop=True)
    sorted_by_nr_laps.insert(0, '#', range(1, len(sorted_by_nr_laps) + 1))
    
    pdf.add_page()
    pdf.add_table(
        sorted_by_nr_laps.values.tolist(),
        [10, 35, 25, 35, 35, 35],
        ['#', 'Transponder ID', 'Total Laps', 'Distance [m]', 'Fastest Lap [s]', 'Avg Lap Time [s]']
    )
    
    # Save the file
    output_path = os.path.join(output_dir, f"rider_report_{group_name}.pdf")
    pdf.output(output_path)



# ------------------------------------------------------------
# 6. Main Execution
# ------------------------------------------------------------
def main():
    # Read in the correct data file 
    csv_file_path = 'RecordingContext_20250214.csv'

    logoURL = "https://idlab.ugent.be/img/logo.png"
    # urllib3.request.urlretrieve(logoURL, "logo.png")
    
    # Example loop positions (unused in this specific code, but available if you need them)
    loop_positions = {
        "L01": 0,
        "L02": 35,
        "L03": 50,
        "L04": 107,
        "L05": 150,
        "L06": 160,
        "L07": 232
    }
    
    # Step 1: Load and preprocess
    df = load_and_preprocess_data(csv_file_path)
    
    # Step 2: Compute metrics
    summary_df, group_stats = compute_metrics(df, track_length=250, loop_filter='L01')
    badman, diesel_engine = general_stats(df)

    df_filtered = df[df['loop'] == 'L01'] if 'loop' in df.columns else df

    # Step 3 & 4: Generate reports for all riders
    all_reports = False
    if all_reports:
        for idx, row in summary_df.iterrows():
            rider_id = row['transponder_id']
            rider_df = df_filtered[df_filtered['transponder_id'] == rider_id]
            # Generate plot
            plot_path_lap_times = generate_lap_time_plot(rider_id, rider_df, group_stats, output_folder='report/plots')
            plot_path_fastest_lap = generate_fastest_lap_comparison_plot(rider_id, summary_df, output_folder='report/plots')
            # For the Speed Over Time plot, pass the ENTIRE df_filtered,
            # so we can show the current rider vs. the rest in gray.
            plot_path_speed_time = generate_speed_over_time_plot(
                rider_id, df_filtered, track_length=250, output_folder='report/plots'
            )
            # Create PDF
            create_rider_pdf_report(rider_id, row, group_stats, plot_path_lap_times,
                                    plot_path_fastest_lap, plot_path_speed_time, output_dir='report',
                                    event_name='IDLab Test Event')
    else:
        # Structure: ['transponder_id','total_laps','total_distance_m','fastest_lap_s','avg_lap_time_s']
        # row[0] gives you the first row, i.e. the first cyclist and his stats
        row = summary_df.iloc[0]
        rider_id = row['transponder_id']
        rider_df = df_filtered[df_filtered['transponder_id'] == rider_id]
        # Generate plot
        plot_path_lap_times = generate_lap_time_plot(rider_id, rider_df, group_stats, output_folder='report/plots')
        plot_path_fastest_lap = generate_fastest_lap_comparison_plot(rider_id, summary_df, output_folder='report/plots')
        # For the Speed Over Time plot, pass the ENTIRE df_filtered,
        # so we can show the current rider vs. the rest in gray.
        plot_path_speed_time = generate_speed_over_time_plot(
            rider_id, df_filtered, track_length=250, output_folder='report/plots'
        )
        # Create PDF
        create_rider_pdf_report(rider_id, row, group_stats, plot_path_lap_times,
                                plot_path_fastest_lap, plot_path_speed_time, output_dir='report',
                                event_name='IDLab Test Event')

        create_general_report('UGent',summary_df,group_stats,badman,diesel_engine,output_dir='report',
                              event_name='IDLab Test Event')
    print("Report generation complete.")

if __name__ == '__main__':
    main()