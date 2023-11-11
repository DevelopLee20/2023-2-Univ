namespace sha256
{
    partial class Form1
    {
        /// <summary>
        /// 필수 디자이너 변수입니다.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 사용 중인 모든 리소스를 정리합니다.
        /// </summary>
        /// <param name="disposing">관리되는 리소스를 삭제해야 하면 true이고, 그렇지 않으면 false입니다.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form 디자이너에서 생성한 코드

        /// <summary>
        /// 디자이너 지원에 필요한 메서드입니다.
        /// 이 메서드의 내용을 코드 편집기로 수정하지 마십시오.
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.Button button1;
            this.txtboxINPUT = new System.Windows.Forms.TextBox();
            this.txtBoxOUPUT = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            button1 = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // button1
            // 
            button1.AutoSize = true;
            button1.Location = new System.Drawing.Point(154, 152);
            button1.Name = "button1";
            button1.Size = new System.Drawing.Size(91, 52);
            button1.TabIndex = 0;
            button1.Text = "SHA-256 변환";
            button1.UseVisualStyleBackColor = true;
            button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // txtboxINPUT
            // 
            this.txtboxINPUT.Location = new System.Drawing.Point(115, 65);
            this.txtboxINPUT.Name = "txtboxINPUT";
            this.txtboxINPUT.Size = new System.Drawing.Size(328, 21);
            this.txtboxINPUT.TabIndex = 1;
            // 
            // txtBoxOUPUT
            // 
            this.txtBoxOUPUT.Location = new System.Drawing.Point(115, 92);
            this.txtBoxOUPUT.Name = "txtBoxOUPUT";
            this.txtBoxOUPUT.Size = new System.Drawing.Size(634, 21);
            this.txtBoxOUPUT.TabIndex = 2;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(71, 68);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(41, 12);
            this.label1.TabIndex = 3;
            this.label1.Text = "입력문";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(71, 92);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(41, 12);
            this.label2.TabIndex = 4;
            this.label2.Text = "암호문";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.GradientActiveCaption;
            this.ClientSize = new System.Drawing.Size(1161, 610);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.txtBoxOUPUT);
            this.Controls.Add(this.txtboxINPUT);
            this.Controls.Add(button1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.TextBox txtboxINPUT;
        private System.Windows.Forms.TextBox txtBoxOUPUT;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
    }
}

