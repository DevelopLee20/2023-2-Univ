namespace LOGIN
{
    partial class Form2
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.panel18 = new System.Windows.Forms.Panel();
            this.button1 = new System.Windows.Forms.Button();
            this.label123 = new System.Windows.Forms.Label();
            this.label122 = new System.Windows.Forms.Label();
            this.label119 = new System.Windows.Forms.Label();
            this.txtPW = new System.Windows.Forms.TextBox();
            this.txtID = new System.Windows.Forms.TextBox();
            this.txtName = new System.Windows.Forms.TextBox();
            this.button5 = new System.Windows.Forms.Button();
            this.button8 = new System.Windows.Forms.Button();
            this.panel18.SuspendLayout();
            this.SuspendLayout();
            // 
            // panel18
            // 
            this.panel18.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel18.Controls.Add(this.button1);
            this.panel18.Controls.Add(this.label123);
            this.panel18.Controls.Add(this.label122);
            this.panel18.Controls.Add(this.label119);
            this.panel18.Controls.Add(this.txtPW);
            this.panel18.Controls.Add(this.txtID);
            this.panel18.Controls.Add(this.txtName);
            this.panel18.Location = new System.Drawing.Point(192, 101);
            this.panel18.Name = "panel18";
            this.panel18.Size = new System.Drawing.Size(416, 248);
            this.panel18.TabIndex = 99;
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(275, 107);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(95, 21);
            this.button1.TabIndex = 10;
            this.button1.Text = "중복확인";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // label123
            // 
            this.label123.AutoSize = true;
            this.label123.Location = new System.Drawing.Point(18, 168);
            this.label123.Name = "label123";
            this.label123.Size = new System.Drawing.Size(59, 12);
            this.label123.TabIndex = 9;
            this.label123.Text = "*비밀번호";
            // 
            // label122
            // 
            this.label122.AutoSize = true;
            this.label122.Location = new System.Drawing.Point(18, 111);
            this.label122.Name = "label122";
            this.label122.Size = new System.Drawing.Size(47, 12);
            this.label122.TabIndex = 3;
            this.label122.Text = "*아이디";
            // 
            // label119
            // 
            this.label119.AutoSize = true;
            this.label119.Location = new System.Drawing.Point(18, 54);
            this.label119.Name = "label119";
            this.label119.Size = new System.Drawing.Size(35, 12);
            this.label119.TabIndex = 2;
            this.label119.Text = "*이름";
            // 
            // txtPW
            // 
            this.txtPW.Location = new System.Drawing.Point(87, 165);
            this.txtPW.Name = "txtPW";
            this.txtPW.PasswordChar = '*';
            this.txtPW.Size = new System.Drawing.Size(147, 21);
            this.txtPW.TabIndex = 3;
            // 
            // txtID
            // 
            this.txtID.Location = new System.Drawing.Point(87, 108);
            this.txtID.Name = "txtID";
            this.txtID.Size = new System.Drawing.Size(147, 21);
            this.txtID.TabIndex = 2;
            this.txtID.TextChanged += new System.EventHandler(this.txtID_TextChanged);
            // 
            // txtName
            // 
            this.txtName.Location = new System.Drawing.Point(87, 51);
            this.txtName.Name = "txtName";
            this.txtName.Size = new System.Drawing.Size(147, 21);
            this.txtName.TabIndex = 1;
            // 
            // button5
            // 
            this.button5.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(95)))), ((int)(((byte)(95)))));
            this.button5.Font = new System.Drawing.Font("HY견고딕", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.button5.ForeColor = System.Drawing.Color.White;
            this.button5.Location = new System.Drawing.Point(406, 383);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(116, 38);
            this.button5.TabIndex = 103;
            this.button5.Text = "확인";
            this.button5.UseVisualStyleBackColor = false;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // button8
            // 
            this.button8.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(166)))), ((int)(((byte)(166)))), ((int)(((byte)(166)))));
            this.button8.Font = new System.Drawing.Font("HY견고딕", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.button8.ForeColor = System.Drawing.Color.White;
            this.button8.Location = new System.Drawing.Point(260, 383);
            this.button8.Name = "button8";
            this.button8.Size = new System.Drawing.Size(116, 38);
            this.button8.TabIndex = 102;
            this.button8.Text = "취소";
            this.button8.UseVisualStyleBackColor = false;
            this.button8.Click += new System.EventHandler(this.button8_Click);
            // 
            // Form2
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.button5);
            this.Controls.Add(this.button8);
            this.Controls.Add(this.panel18);
            this.Name = "Form2";
            this.Text = "Form2";
            this.panel18.ResumeLayout(false);
            this.panel18.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel18;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Label label123;
        private System.Windows.Forms.Label label122;
        private System.Windows.Forms.Label label119;
        private System.Windows.Forms.TextBox txtPW;
        private System.Windows.Forms.TextBox txtID;
        private System.Windows.Forms.TextBox txtName;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.Button button8;
    }
}