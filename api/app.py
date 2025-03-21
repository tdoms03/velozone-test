from flask import Flask, render_template
from app.extra_functions import limit_numeric_to_2_decimals
from app.data_analysis_classes import DataAnalysis
from app.Read_supabase_data import *
import pandas as pd
import json
import os


PER_PAGE = 10  # Number of riders per page

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/leaderboard')
def leaderboard():
    # Data fetch




    return render_template('leaderboard.html')#, 
                            #averages=avg_lap, 
                            #top_laps=fast_lap, 
                            #slow_lap=slow_lap, 
                            #badman_lap=badman,
                            #diesel=diesel,
                            #electric=electric)

    #return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)