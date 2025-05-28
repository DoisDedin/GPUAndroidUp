package com.example.vulkanfft

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.vulkanfft.view.MainActivityModule

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val intent = Intent(this, MainActivityModule::class.java)
        startActivity(intent)
        Log.d("MainActivity", "OPEN")
    }
}