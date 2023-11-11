namespace LOGIN
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
        /// 이 메서드의 내용을 코드 편집기로 수정하지 마세요.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.imageList1 = new System.Windows.Forms.ImageList(this.components);
            this.imageList2 = new System.Windows.Forms.ImageList(this.components);
            this.imageList3 = new System.Windows.Forms.ImageList(this.components);
            this.loginPanel = new System.Windows.Forms.Panel();
            this.label4 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.button24 = new System.Windows.Forms.Button();
            this.labelError = new System.Windows.Forms.Label();
            this.loginlabel = new System.Windows.Forms.Label();
            this.loginSignupbutton = new System.Windows.Forms.Button();
            this.login = new System.Windows.Forms.Button();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.loginPanel.SuspendLayout();
            this.SuspendLayout();
            // 
            // imageList1
            // 
            this.imageList1.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageList1.ImageStream")));
            this.imageList1.TransparentColor = System.Drawing.Color.Transparent;
            this.imageList1.Images.SetKeyName(0, "노이미지.png");
            // 
            // imageList2
            // 
            this.imageList2.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageList2.ImageStream")));
            this.imageList2.TransparentColor = System.Drawing.Color.Transparent;
            this.imageList2.Images.SetKeyName(0, "1.png");
            this.imageList2.Images.SetKeyName(1, "2.png");
            this.imageList2.Images.SetKeyName(2, "3.png");
            this.imageList2.Images.SetKeyName(3, "4.png");
            this.imageList2.Images.SetKeyName(4, "5.png");
            this.imageList2.Images.SetKeyName(5, "6.png");
            // 
            // imageList3
            // 
            this.imageList3.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageList3.ImageStream")));
            this.imageList3.TransparentColor = System.Drawing.Color.Transparent;
            this.imageList3.Images.SetKeyName(0, "인덱스1.PNG");
            this.imageList3.Images.SetKeyName(1, "인덱스2.PNG");
            this.imageList3.Images.SetKeyName(2, "인덱스3.PNG");
            this.imageList3.Images.SetKeyName(3, "인덱스4.PNG");
            // 
            // loginPanel
            // 
            this.loginPanel.Controls.Add(this.label4);
            this.loginPanel.Controls.Add(this.label2);
            this.loginPanel.Controls.Add(this.button24);
            this.loginPanel.Controls.Add(this.labelError);
            this.loginPanel.Controls.Add(this.loginlabel);
            this.loginPanel.Controls.Add(this.loginSignupbutton);
            this.loginPanel.Controls.Add(this.login);
            this.loginPanel.Controls.Add(this.textBox2);
            this.loginPanel.Controls.Add(this.textBox1);
            this.loginPanel.Location = new System.Drawing.Point(202, 105);
            this.loginPanel.Name = "loginPanel";
            this.loginPanel.Size = new System.Drawing.Size(397, 309);
            this.loginPanel.TabIndex = 68;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("HY견고딕", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label4.ForeColor = System.Drawing.Color.Silver;
            this.label4.Location = new System.Drawing.Point(23, 86);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(41, 19);
            this.label4.TabIndex = 17;
            this.label4.Text = "PW";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("HY견고딕", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label2.ForeColor = System.Drawing.Color.Silver;
            this.label2.Location = new System.Drawing.Point(33, 48);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(31, 19);
            this.label2.TabIndex = 16;
            this.label2.Text = "ID";
            // 
            // button24
            // 
            this.button24.BackColor = System.Drawing.Color.Silver;
            this.button24.Font = new System.Drawing.Font("굴림", 9F, System.Drawing.FontStyle.Bold);
            this.button24.ForeColor = System.Drawing.Color.White;
            this.button24.Location = new System.Drawing.Point(247, 211);
            this.button24.Name = "button24";
            this.button24.Size = new System.Drawing.Size(129, 36);
            this.button24.TabIndex = 15;
            this.button24.Text = "돌아가기";
            this.button24.UseVisualStyleBackColor = false;
            this.button24.Click += new System.EventHandler(this.button24_Click);
            // 
            // labelError
            // 
            this.labelError.AutoSize = true;
            this.labelError.Location = new System.Drawing.Point(68, 127);
            this.labelError.Name = "labelError";
            this.labelError.Size = new System.Drawing.Size(81, 12);
            this.labelError.TabIndex = 12;
            this.labelError.Text = "비밀번호 확인";
            // 
            // loginlabel
            // 
            this.loginlabel.AutoSize = true;
            this.loginlabel.Font = new System.Drawing.Font("HY견고딕", 18F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.loginlabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(95)))), ((int)(((byte)(95)))));
            this.loginlabel.Location = new System.Drawing.Point(20, 12);
            this.loginlabel.Name = "loginlabel";
            this.loginlabel.Size = new System.Drawing.Size(82, 24);
            this.loginlabel.TabIndex = 10;
            this.loginlabel.Text = "로그인";
            // 
            // loginSignupbutton
            // 
            this.loginSignupbutton.BackColor = System.Drawing.Color.Silver;
            this.loginSignupbutton.Font = new System.Drawing.Font("굴림", 9F, System.Drawing.FontStyle.Bold);
            this.loginSignupbutton.ForeColor = System.Drawing.Color.White;
            this.loginSignupbutton.Location = new System.Drawing.Point(70, 211);
            this.loginSignupbutton.Name = "loginSignupbutton";
            this.loginSignupbutton.Size = new System.Drawing.Size(129, 36);
            this.loginSignupbutton.TabIndex = 9;
            this.loginSignupbutton.Text = "회원가입";
            this.loginSignupbutton.UseVisualStyleBackColor = false;
            this.loginSignupbutton.Click += new System.EventHandler(this.loginSignupbutton_Click_1);
            // 
            // login
            // 
            this.login.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(95)))), ((int)(((byte)(95)))));
            this.login.Font = new System.Drawing.Font("HY견고딕", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.login.ForeColor = System.Drawing.Color.White;
            this.login.Location = new System.Drawing.Point(70, 142);
            this.login.Name = "login";
            this.login.Size = new System.Drawing.Size(306, 45);
            this.login.TabIndex = 6;
            this.login.Text = "로그인";
            this.login.UseVisualStyleBackColor = false;
            this.login.Click += new System.EventHandler(this.login_Click_1);
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(70, 86);
            this.textBox2.Name = "textBox2";
            this.textBox2.PasswordChar = '*';
            this.textBox2.Size = new System.Drawing.Size(306, 21);
            this.textBox2.TabIndex = 5;
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(70, 48);
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(306, 21);
            this.textBox1.TabIndex = 4;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(968, 582);
            this.Controls.Add(this.loginPanel);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "Form1";
            this.Text = "로그인";
            this.loginPanel.ResumeLayout(false);
            this.loginPanel.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.ImageList imageList1;
        private System.Windows.Forms.ImageList imageList2;
        private System.Windows.Forms.ImageList imageList3;
        private System.Windows.Forms.Panel loginPanel;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button button24;
        private System.Windows.Forms.Label labelError;
        private System.Windows.Forms.Label loginlabel;
        private System.Windows.Forms.Button loginSignupbutton;
        private System.Windows.Forms.Button login;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.TextBox textBox1;
    }
}

