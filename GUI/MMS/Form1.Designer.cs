
using System.Windows.Forms;
namespace MusicGenreClassifier
{
    partial class Form1
    {
        private System.ComponentModel.IContainer components = null;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        private void InitializeComponent()
        {
            this.txtFilePath = new System.Windows.Forms.TextBox();
            this.btnBrowse = new System.Windows.Forms.Button();
            this.btnPlayPause = new System.Windows.Forms.Button();
            this.btnPredict = new System.Windows.Forms.Button();
            this.lblGenre = new System.Windows.Forms.Label();
            this.comboBoxModel = new System.Windows.Forms.ComboBox();
            this.buttonPredictAll = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // txtFilePath
            // 
            this.txtFilePath.Location = new System.Drawing.Point(16, 18);
            this.txtFilePath.Margin = new System.Windows.Forms.Padding(4);
            this.txtFilePath.Name = "txtFilePath";
            this.txtFilePath.ReadOnly = true;
            this.txtFilePath.Size = new System.Drawing.Size(399, 22);
            this.txtFilePath.TabIndex = 0;
            // 
            // btnBrowse
            // 
            this.btnBrowse.Location = new System.Drawing.Point(424, 15);
            this.btnBrowse.Margin = new System.Windows.Forms.Padding(4);
            this.btnBrowse.Name = "btnBrowse";
            this.btnBrowse.Size = new System.Drawing.Size(100, 28);
            this.btnBrowse.TabIndex = 1;
            this.btnBrowse.Text = "Odaberi";
            this.btnBrowse.UseVisualStyleBackColor = true;
            this.btnBrowse.Click += new System.EventHandler(this.btnBrowse_Click);
            // 
            // btnPlayPause
            // 
            this.btnPlayPause.Location = new System.Drawing.Point(16, 62);
            this.btnPlayPause.Margin = new System.Windows.Forms.Padding(4);
            this.btnPlayPause.Name = "btnPlayPause";
            this.btnPlayPause.Size = new System.Drawing.Size(100, 28);
            this.btnPlayPause.TabIndex = 2;
            this.btnPlayPause.Text = "Play";
            this.btnPlayPause.UseVisualStyleBackColor = true;
            this.btnPlayPause.Click += new System.EventHandler(this.btnPlayPause_Click);
            // 
            // btnPredict
            // 
            this.btnPredict.Location = new System.Drawing.Point(424, 62);
            this.btnPredict.Margin = new System.Windows.Forms.Padding(4);
            this.btnPredict.Name = "btnPredict";
            this.btnPredict.Size = new System.Drawing.Size(100, 28);
            this.btnPredict.TabIndex = 3;
            this.btnPredict.Text = "Predvidi";
            this.btnPredict.UseVisualStyleBackColor = true;
            this.btnPredict.Click += new System.EventHandler(this.btnPredict_Click);
            // 
            // lblGenre
            // 
            this.lblGenre.AutoSize = true;
            this.lblGenre.Location = new System.Drawing.Point(13, 112);
            this.lblGenre.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.lblGenre.Name = "lblGenre";
            this.lblGenre.Size = new System.Drawing.Size(106, 16);
            this.lblGenre.TabIndex = 4;
            this.lblGenre.Text = "Predviđeni žanr: ";
            // 
            // comboBoxModel
            // 
            this.comboBoxModel.FormattingEnabled = true;
            this.comboBoxModel.Location = new System.Drawing.Point(267, 62);
            this.comboBoxModel.Name = "comboBoxModel";
            this.comboBoxModel.Size = new System.Drawing.Size(148, 24);
            this.comboBoxModel.TabIndex = 5;
            // 
            // buttonPredictAll
            // 
            this.buttonPredictAll.Location = new System.Drawing.Point(123, 62);
            this.buttonPredictAll.Name = "buttonPredictAll";
            this.buttonPredictAll.Size = new System.Drawing.Size(100, 28);
            this.buttonPredictAll.TabIndex = 6;
            this.buttonPredictAll.Text = "Predvidi sve";
            this.buttonPredictAll.UseVisualStyleBackColor = true;
            this.buttonPredictAll.Click += new System.EventHandler(this.buttonPredictAll_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(539, 244);
            this.Controls.Add(this.buttonPredictAll);
            this.Controls.Add(this.comboBoxModel);
            this.Controls.Add(this.lblGenre);
            this.Controls.Add(this.btnPredict);
            this.Controls.Add(this.btnPlayPause);
            this.Controls.Add(this.btnBrowse);
            this.Controls.Add(this.txtFilePath);
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "Form1";
            this.Text = "Klasifikator glazbenih žanrova";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private TextBox txtFilePath;
        private Button btnBrowse;
        private Button btnPlayPause;
        private Button btnPredict;
        private Label lblGenre;
        private ComboBox comboBoxModel;
        private Button buttonPredictAll;
    }
}
